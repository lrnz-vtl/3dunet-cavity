import os
import shutil
from multiprocessing import Lock
from pathlib import Path
import h5py
import numpy as np
from openbabel import openbabel
from potsim2 import PotGrid
import prody as pr
import subprocess
from scipy.spatial.transform import Rotation
import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.augment.transforms import SampleStats
import pytorch3dunet.augment.featurizer as featurizer
from pytorch3dunet.augment.featurizer import Grid
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats, sample_instances, \
    default_prediction_collate
from pytorch3dunet.unet3d.utils import get_logger, profile
import uuid

from multiprocessing import Pool, cpu_count

logger = get_logger('HDF5Dataset')
lock = Lock()


def apbsInput(pqr_fname, grid_fname, dielec_const=4.0, grid_size=161):
    return f"""read
    mol pqr {pqr_fname}
end
elec name prot
    mg-manual
    mol 1
    dime {grid_size} {grid_size} {grid_size}
    grid 1.0 1.0 1.0
    gcent mol 1
    lpbe
    bcfl mdh
    ion charge 1 conc 0.100 radius 2.0
    ion charge -1 conc 0.100 radius 2.0
    pdie {dielec_const}
    sdie 78.54
    sdens 10.0
    chgm spl2
    srfm smol
    srad 0.0
    swin 0.3
    temp 298.15
    calcenergy total
    calcforce no
    write pot gz {grid_fname}
end"""


class DataPaths:
    def __init__(self, h5_path, pdb_path=None, pocket_path=None, grid_path=None):
        self.h5_path = h5_path

        if pdb_path is not None and os.path.exists(pdb_path):
            self.pdb_path = str(pdb_path)
        else:
            self.pdb_path = None
        if pocket_path is not None and os.path.exists(pocket_path):
            self.pocket_path = str(pocket_path)
        else:
            self.pocket_path = None
        if grid_path is not None and os.path.exists(grid_path):
            self.grid_path = str(grid_path)
        else:
            self.grid_path = None


class AbstractDataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        raise NotImplementedError

    def __init__(self, raws, labels, weight_maps, name, tmp_data_folder,
                 phase,
                 slice_builder_config,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
                 instance_ratio=None,
                 random_seed=0):
        """
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :param slice_builder_config: configuration of the SliceBuilder
        :param transformer_config: data augmentation configuration
        :param mirror_padding (int or tuple): number of voxels padded to each axis
        :param a number between (0, 1]: specifies a fraction of ground truth instances to be sampled from the dense ground truth labels
        """

        self.name = name
        self.tmp_data_folder = tmp_data_folder

        self.h5path = Path(self.tmp_data_folder) / f'grids.h5'

        self.hasWeights = weight_maps is not None

        assert phase in ['train', 'val', 'test']
        if phase in ['train', 'val']:
            mirror_padding = None

        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"

        self.mirror_padding = mirror_padding
        self.phase = phase

        self.instance_ratio = instance_ratio

        self.stats = SampleStats(raws)
        # min_value, max_value, mean, std = self.ds_stats(raws)

        self.transformer = transforms.get_transformer(transformer_config, stats=self.stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase != 'test':
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.label_transform()

            if self.instance_ratio is not None:
                assert 0 < self.instance_ratio <= 1
                rs = np.random.RandomState(random_seed)
                labels = [sample_instances(m, self.instance_ratio, rs) for m in labels]

            if self.hasWeights:
                self.weight_transform = self.transformer.weight_transform()

            self._check_dimensionality(raws, labels)
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            labels = None

            # add mirror padding if needed
            if self.mirror_padding is not None:
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                padded_volumes = []
                for raw in raws:
                    if raw.ndim == 4:
                        channels = [np.pad(r, pad_width=pad_width, mode='reflect') for r in raw]
                        padded_volume = np.stack(channels)
                    else:
                        padded_volume = np.pad(raw, pad_width=pad_width, mode='reflect')

                    padded_volumes.append(padded_volume)

                raws = padded_volumes

        # build slice indices for raw and label data sets
        slice_builder = get_slice_builder(raws, labels, weight_maps, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices

        # TODO Only cache to file for validation dataset
        with h5py.File(self.h5path, 'w') as h5:
            h5.create_dataset('raws', data=raws)
            if labels is not None:
                h5.create_dataset('labels', data=labels)
            if weight_maps is not None:
                h5.create_dataset('weight_maps', data=weight_maps)

        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    def ds_stats(self, raws):
        # calculate global min, max, mean and std for normalization
        min_value, max_value, mean, std = calculate_stats(raws)
        logger.info(f'Input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')
        return min_value, max_value, mean, std

    def __getitem__(self, idx):

        logger.info(f'Getting idx {idx} from {self.name}')

        if idx >= len(self):
            raise StopIteration

        # TODO This is inefficient with patches
        with h5py.File(self.h5path, 'r') as h5:
            raws = list(h5['raws'])

            # get the slice for a given index 'idx'
            raw_idx = self.raw_slices[idx]
            # get the raw data patch for a given slice
            raw_patch_transformed = self._transform_patches(raws, raw_idx, self.raw_transform)

            if self.phase == 'test':
                # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
                if len(raw_idx) == 4:
                    raw_idx = raw_idx[1:]
                return self.name, (raw_patch_transformed, raw_idx)
            else:
                labels = list(h5['labels'])
                # get the slice for a given index 'idx'
                label_idx = self.label_slices[idx]
                label_patch_transformed = self._transform_patches(labels, label_idx, self.label_transform)
                if self.hasWeights:
                    weight_maps = h5['weight_maps']
                    weight_idx = self.weight_slices[idx]
                    # return the transformed weight map for a given patch together with raw and label data
                    weight_patch_transformed = self._transform_patches(weight_maps, weight_idx, self.weight_transform)
                    return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
                # return the transformed raw and label patches
                return self.name, (raw_patch_transformed, label_patch_transformed)

    @classmethod
    def prediction_collate(cls, batch):
        names = [name for name,_ in batch]
        assert all(x == names[0] for x in names)
        samples = [data for _,data in batch]
        return default_prediction_collate(samples)

    @staticmethod
    def _transform_patches(datasets, label_idx, transformer):
        transformed_patches = []
        for dataset in datasets:
            # get the label data and apply the label transformer
            transformed_patch = transformer(dataset[label_idx])
            transformed_patches.append(transformed_patch)

        # if transformed_patches is a singleton list return the first element only
        if len(transformed_patches) == 1:
            return transformed_patches[0]
        else:
            return transformed_patches

    def __len__(self):
        return self.patch_count

    @staticmethod
    def _check_dimensionality(raws, labels):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        for raw, label in zip(raws, labels):
            assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

            assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'


class StandardPDBDataset(AbstractDataset):

    def __init__(self, src_data_folder, name, exe_config,
                 phase,
                 slice_builder_config,
                 pregrid_transformer_config,
                 grid_config,
                 features_config,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
                 instance_ratio=None,
                 random_seed=0):
        self.src_data_folder = src_data_folder
        self.name = name
        self.cleanup = exe_config.get('cleanup', False)

        uuids = uuid.uuid1()
        self.tmp_data_folder = str(Path(exe_config['tmp_folder']) / f"{name}_{uuids}")
        os.makedirs(self.tmp_data_folder)

        self.pdb2pqrPath = exe_config['pdb2pqrPath']
        self.dielec_const = grid_config.get('dielec_const', 4.0)
        self.grid_size = grid_config.get('grid_size', 161)

        assert phase in ['train', 'val', 'test']

        try:
            structure, ligand = self._processPdb()

            for elem in pregrid_transformer_config:
                if elem['name'] == 'RandomRotate':
                    structure, ligand = self._randomRotatePdb(structure, ligand, self.name)

            pot_grid, labels = self._genGrids(structure, ligand)
            # Serialise structure
            self.structure_fname = str(Path(self.tmp_data_folder) / "structure.pdb")
            pr.writePDB(self.structure_fname, structure)

        except Exception as e:
            raise type(e)(f"Tmp folder: {self.tmp_data_folder}") from e

        self.grid = Grid(pot_grid, self.grid_size)
        labels = self.grid.homologate_labels(labels)

        features = featurizer.get_featurizer(features_config).raw_transform()
        raws = features(structure, self.grid)
        self.grid.delGrid()

        self.ndim = raws.ndim
        self.shape = raws.shape

        raws = [raws]
        if phase == 'test':
            # create label/weight transform only in train/val phase
            labels = None
        else:
            labels = [labels]

        weight_maps = None

        super().__init__(raws, labels, weight_maps,
                         name=self.name,
                         tmp_data_folder=self.tmp_data_folder,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         mirror_padding=mirror_padding,
                         instance_ratio=instance_ratio,
                         random_seed=random_seed)

    def getStructure(self):
        return pr.parsePDB(self.structure_fname)

    # def __del__(self):
    #     shutil.rmtree(self.tmp_data_folder)

    def _remove(self,fname):
        if self.cleanup:
            os.remove(fname)

    def _processPdb(self):

        src_pdb_file = f'{self.src_data_folder}/{self.name}/{self.name}_protein.pdb'
        src_mol_file = f"{self.src_data_folder}/{self.name}/{self.name}_ligand.mol2"
        tmp_ligand_pdb_file = str(Path(self.tmp_data_folder) / f'ligand.pdb')

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("mol2", "pdb")

        # remove water molecules (found some pdb files with water molecules)
        structure = pr.parsePDB(src_pdb_file)
        protein = structure.select('protein').toAtomGroup()

        # convert ligand to pdb
        ligand = openbabel.OBMol()

        obConversion.ReadFile(ligand, src_mol_file)
        obConversion.WriteFile(ligand, tmp_ligand_pdb_file)

        # select only chains that are close to the ligand (I love ProDy v2)
        ligand = pr.parsePDB(tmp_ligand_pdb_file)
        lresname = ligand.getResnames()[0]
        complx = ligand + protein

        # select ONLY atoms that belong to the protein
        complx = complx.select(f'same chain as exwithin 7 of resname {lresname}')
        complx = complx.select(f'protein and not resname {lresname}')

        self._remove(tmp_ligand_pdb_file)

        return complx, ligand

    @profile
    def _runApbs(self, dst_pdb_file):
        pqr_output = f"{self.tmp_data_folder}/protein.pqr"
        grid_fname = f"{self.tmp_data_folder}/grid.dx.gz"

        logger.debug(f'Running pdb2pqr on {self.name}')

        proc = subprocess.Popen(
            [
                self.pdb2pqrPath,
                "--with-ph=7.4",
                "--ff=PARSE",
                "--chain",
                dst_pdb_file,
                pqr_output
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        cmd_out = proc.communicate()
        if proc.returncode != 0:
            raise Exception(cmd_out[1].decode())

        logger.debug(f'Running apbs on {self.name}')

        owd = os.getcwd()
        os.chdir(self.tmp_data_folder)

        apbs_in_fname = "apbs-in"
        input = apbsInput(pqr_fname=str(Path(pqr_output).name), grid_fname=str(Path(grid_fname).name).split('.')[0],
                          grid_size=self.grid_size, dielec_const=self.dielec_const)

        with open(apbs_in_fname, "w") as f:
            f.write(input)

        # generates dx.gz grid file
        proc = subprocess.Popen(
            ["apbs", apbs_in_fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        cmd_out = proc.communicate()
        if proc.returncode != 0:
            raise Exception(cmd_out[1].decode())

        self._remove(apbs_in_fname)
        os.chdir(owd)

        self._remove(pqr_output)

        grid = PotGrid(dst_pdb_file, grid_fname)
        self._remove(grid_fname)

        return grid

    def _genGrids(self, structure, ligand):

        dst_pdb_file = f'{self.tmp_data_folder}/protein_trans.pdb'

        pr.writePDB(dst_pdb_file, structure)
        # pdb2pqr fails to read pdbs with the one line header generated by ProDy...
        with open(dst_pdb_file, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(dst_pdb_file, 'w') as fout:
            fout.writelines(data[1:])

        grid = self._runApbs(dst_pdb_file)
        self._remove(dst_pdb_file)

        # ligand mask is a boolean NumPy array, can be converted to int: ligand_mask.astype(int)
        ligand_mask = grid.get_ligand_mask(ligand)

        return grid, ligand_mask

    @classmethod
    def _randomRotatePdb(cls, structure, ligand, name=None):
        # Todo Init seed?
        m = Rotation.random()
        # m = Rotation.from_euler(angles=[0.29980811330064344, 0.3443362966037462, 2.2242614439106614], seq='zxy')

        r = m.as_matrix()

        angles = m.as_euler('zxy')
        logger.info(f'Random rotation matrix angles: {list(angles)} to {name}')

        coords_prot = np.einsum('ij,kj->ki', r, structure.getCoords())
        coords_ligand = np.einsum('ij,kj->ki', r, ligand.getCoords())

        coords_prot_mean = coords_prot.mean(axis=0)

        # Subtract center of mass
        coords_prot = coords_prot - coords_prot_mean
        coords_ligand = coords_ligand - coords_prot_mean

        structure2 = structure.copy()
        structure2.setCoords(coords_prot)
        ligand2 = ligand.copy()
        ligand2.setCoords(coords_ligand)
        return structure2, ligand2

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        logger.info(f"Slice builder config: {phase_config['slice_builder']}")

        file_paths = phase_config['file_paths']
        file_paths = cls.traverse_pdb_paths(file_paths)

        args = [(file_path, name, dataset_config, phase) for file_path, name in file_paths]

        if dataset_config.get('parallel', True):
            nworkers = min(cpu_count(), max(1, dataset_config.get('num_workers', 1)))
            logger.info(f'Parallelizing dataset creation among {nworkers} workers')
            pool = Pool(processes=nworkers)
            return [x for x in pool.map(create_dataset, args) if x is not None]
        else:
            return [x for x in (create_dataset(arg) for arg in args) if x is not None]

    @staticmethod
    def traverse_pdb_paths(file_paths):

        assert isinstance(file_paths, list)
        results = []

        for file_path in file_paths:
            file_path = Path(file_path)
            name = str(file_path.name)

            pdbfile = file_path / f"{name}_protein.pdb"
            molfile = file_path / f"{name}_ligand.mol2"

            if os.path.exists(pdbfile) and os.path.exists(molfile):
                results.append((file_path.parent, name))

        return results


def create_dataset(arg):
    file_path, name, dataset_config, phase = arg
    phase_config = dataset_config[phase]

    # load data augmentation configuration
    transformer_config = phase_config['transformer']
    pregrid_transformer_config = phase_config.get('pdb_transformer', [])
    grid_config = dataset_config.get('grid_config', {})
    features_config = dataset_config.get('featurizer', [])

    # load slice builder config
    slice_builder_config = phase_config['slice_builder']

    # load instance sampling configuration
    instance_ratio = phase_config.get('instance_ratio', None)
    random_seed = phase_config.get('random_seed', 0)

    exe_config = {k: dataset_config[k] for k in ['tmp_folder', 'pdb2pqrPath', 'cleanup'] if k in dataset_config.keys()}

    try:
        logger.info(f'Loading {phase} set from: {file_path} named {name} ...')
        dataset = StandardPDBDataset(src_data_folder=file_path,
                                     name=name,
                                     exe_config=exe_config,
                                     phase=phase,
                                     slice_builder_config=slice_builder_config,
                                     features_config=features_config,
                                     transformer_config=transformer_config,
                                     pregrid_transformer_config=pregrid_transformer_config,
                                     grid_config=grid_config,
                                     mirror_padding=dataset_config.get('mirror_padding', None),
                                     instance_ratio=instance_ratio,
                                     random_seed=random_seed)
        return dataset
    except Exception:
        logger.error(f'Skipping {phase} set from: {file_path} named {name}.', exc_info=True)
        return None