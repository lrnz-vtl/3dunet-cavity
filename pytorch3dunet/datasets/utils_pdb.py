import os
from pathlib import Path
from openbabel import openbabel
import prody as pr
from pytorch3dunet.unet3d.utils import get_logger
import subprocess
import pytorch3dunet.augment.featurizer as featurizer
from pytorch3dunet.augment.featurizer import Grid
from scipy.spatial.transform import Rotation
from multiprocessing import Pool, cpu_count
import numpy as np
from potsim2 import PotGrid
import uuid

logger = get_logger('UtilsPdb')

dielec_const_default = 4.0
grid_size_default = 161
ligand_mask_radius_defaults = 6.5

def apbsInput(pqr_fname, grid_fname, dielec_const, grid_size):
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

def processPdb(src_data_folder, name, pdb_ligand_fname):
    """ Generate pdb training data from raw data """

    src_pdb_file = f'{src_data_folder}/{name}/{name}_protein.pdb'
    src_mol_file = f"{src_data_folder}/{name}/{name}_ligand.mol2"

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol2", "pdb")

    # remove water molecules (found some pdb files with water molecules)
    structure = pr.parsePDB(src_pdb_file)
    protein = structure.select('protein').toAtomGroup()

    # convert ligand to pdb
    ligand = openbabel.OBMol()

    obConversion.ReadFile(ligand, src_mol_file)
    obConversion.WriteFile(ligand, pdb_ligand_fname)

    # select only chains that are close to the ligand (I love ProDy v2)
    ligand = pr.parsePDB(pdb_ligand_fname)
    lresname = ligand.getResnames()[0]
    complx = ligand + protein

    # select ONLY atoms that belong to the protein
    complx = complx.select(f'same chain as exwithin 7 of resname {lresname}')
    complx = complx.select(f'protein and not resname {lresname}')

    return complx, ligand


class PdbDataHandler:
    def __init__(self,
                 src_data_folder,
                 name,
                 pregrid_transformer_config,
                 tmp_data_folder,
                 pdb2pqrPath,
                 cleanup):

        self.src_data_folder = src_data_folder
        self.name = name
        self.tmp_data_folder = tmp_data_folder
        self.pdb2pqrPath = pdb2pqrPath
        self.cleanup = cleanup
        self.grid = None

        structure, ligand = self._processPdb()

        for elem in pregrid_transformer_config:
            if elem['name'] == 'RandomRotate':
                structure, ligand = self._randomRotatePdb(structure, ligand)

        # Serialise structure to file, needed for pickling
        self.structure_fname = str(Path(self.tmp_data_folder) / "structure.pdb")
        self.ligand_fname = str(Path(self.tmp_data_folder) / "ligand.pdb")
        pr.writePDB(self.structure_fname, structure)
        pr.writePDB(self.ligand_fname, ligand)

    def getStructureLigand(self):
        return pr.parsePDB(self.structure_fname), pr.parsePDB(self.ligand_fname)

    def getRawsLabels(self, features_config, grid_config):
        ''' This also populates the self.grid variable '''
        grid_size = grid_config.get('grid_size', grid_size_default)

        structure, ligand = self.getStructureLigand()

        pot_grid, labels = self._genGrids(structure, ligand, grid_config)

        self.grid = Grid(pot_grid, grid_size)
        labels = self.grid.homologate_labels(labels)

        features = featurizer.get_featurizer(features_config).raw_transform()
        raws = features(structure, self.grid)
        self.grid.delGrid()
        return raws, labels

    def _randomRotatePdb(self, structure, ligand):
        # Todo Init seed?
        m = Rotation.random()
        # m = Rotation.from_euler(angles=[0.29980811330064344, 0.3443362966037462, 2.2242614439106614], seq='zxy')

        r = m.as_matrix()

        angles = m.as_euler('zxy')
        logger.debug(f'Random rotation matrix angles: {list(angles)} to {self.name}')

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

    def _remove(self,fname):
        if self.cleanup:
            os.remove(fname)

    def _processPdb(self):
        tmp_ligand_pdb_file = str(Path(self.tmp_data_folder) / f'ligand_tmp.pdb')
        complx, ligand = processPdb(self.src_data_folder, self.name, tmp_ligand_pdb_file)
        self._remove(tmp_ligand_pdb_file)
        return complx, ligand

    def _runApbs(self, dst_pdb_file, grid_size, dielec_const):
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
                          grid_size=grid_size, dielec_const=dielec_const)

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

    def _genGrids(self, structure, ligand, grid_config):

        dielec_const = grid_config.get('dielec_const', dielec_const_default)
        radius = grid_config.get('ligand_mask_radius', ligand_mask_radius_defaults)
        grid_size = grid_config.get('grid_size', grid_size_default)

        dst_pdb_file = f'{self.tmp_data_folder}/protein_trans.pdb'

        pr.writePDB(dst_pdb_file, structure)
        # pdb2pqr fails to read pdbs with the one line header generated by ProDy...
        with open(dst_pdb_file, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(dst_pdb_file, 'w') as fout:
            fout.writelines(data[1:])

        grid = self._runApbs(dst_pdb_file, grid_size=grid_size, dielec_const=dielec_const)
        self._remove(dst_pdb_file)

        # ligand mask is a boolean NumPy array, can be converted to int: ligand_mask.astype(int)
        ligand_mask = grid.get_ligand_mask(ligand, radius)

        return grid, ligand_mask

    @classmethod
    def map_datasets(cls, dataset_config, phase, f):
        phase_config = dataset_config[phase]

        file_paths = phase_config['file_paths']
        file_paths = cls.traverse_pdb_paths(file_paths)

        args = [(file_path, name, dataset_config, phase) for file_path, name in file_paths]

        if dataset_config.get('parallel', True):
            nworkers = min(cpu_count(), max(1, dataset_config.get('num_workers', 1)))
            logger.info(f'Parallelizing dataset creation among {nworkers} workers')
            pool = Pool(processes=nworkers)
            return [f(*x) for x in pool.map(create_dataset, args) if x is not None]
        else:
            return [f(*x) for x in (create_dataset(arg) for arg in args) if x is not None]

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

    pregrid_transformer_config = phase_config.get('pdb_transformer', [])
    grid_config = dataset_config.get('grid_config', {})
    features_config = dataset_config.get('featurizer', [])

    exe_config = {k: dataset_config[k] for k in ['tmp_folder', 'pdb2pqrPath', 'cleanup'] if
                  k in dataset_config.keys()}

    tmp_data_folder = str(Path(exe_config['tmp_folder']) / name)
    os.makedirs(tmp_data_folder, exist_ok=True)

    try:
        logger.info(f'Loading {phase} set from: {file_path} named {name} ...')
        dataset = PdbDataHandler(src_data_folder=file_path,
                                 name=name,
                                 pregrid_transformer_config=pregrid_transformer_config,
                                 tmp_data_folder=tmp_data_folder,
                                 pdb2pqrPath=exe_config['pdb2pqrPath'],
                                 cleanup=exe_config.get('cleanup', False))
        raws,labels = dataset.getRawsLabels(features_config=features_config, grid_config=grid_config)
        return raws, labels

    except Exception:
        logger.error(f'Skipping {phase} set from: {file_path} named {name}.', exc_info=True)
        return None