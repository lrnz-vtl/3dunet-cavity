import os
from pathlib import Path
from multiprocessing import Lock, Pool
import h5py
import numpy as np
import glob
import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.augment.transforms import SampleStats
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats, sample_instances, \
    default_prediction_collate
from pytorch3dunet.datasets.utils_pdb import PdbDataHandler
from pytorch3dunet.unet3d.utils import get_logger
import uuid
from torch.utils.data._utils.collate import default_collate
from pytorch3dunet.datasets.featurizer import BaseFeatureList, get_features

logger = get_logger('PdbDataset')
lock = Lock()


class AbstractDataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, raws, labels, weight_maps,
                 featuresTypes,
                 tmp_data_folder,
                 phase,
                 slice_builder_config,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
                 instance_ratio=None,
                 random_seed=0,
                 allowRotations=True,
                 debug_str=None):
        """
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :param slice_builder_config: configuration of the SliceBuilder
        :param transformer_config: data augmentation configuration
        :param mirror_padding (int or tuple): number of voxels padded to each axis
        :param a number between (0, 1]: specifies a fraction of ground truth instances to be sampled from the dense ground truth labels
        """
        self.tmp_data_folder = tmp_data_folder
        self.h5path = Path(self.tmp_data_folder) / f'grids.h5'
        self.hasWeights = weight_maps is not None
        self.featureTypes = featuresTypes

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

        self.transformer = transforms.get_transformer(transformer_config, allowRotations=allowRotations,
                                                      debug_str=debug_str, stats=self.stats)
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
        if idx >= len(self):
            raise StopIteration

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
                return (raw_patch_transformed, raw_idx)
            else:
                labels = list(np.array(h5['labels']).astype("<f8"))
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
                return (raw_patch_transformed, label_patch_transformed)

    def _transform_patches(self, datasets, label_idx, transformer):
        transformed_patches = []
        for dataset in datasets:
            # get the label data and apply the label transformer
            transformed_patch = transformer(dataset[label_idx], self.featuresTypes)
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
                 features: BaseFeatureList,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
                 instance_ratio=None,
                 random_seed=0):

        reuse_grids = exe_config.get('reuse_grids', False)
        randomize_name = exe_config.get('randomize_name', False)

        self.src_data_folder = src_data_folder
        self.name = name

        assert phase in ['train', 'val', 'test']

        if randomize_name:
            if reuse_grids:
                existing = list(glob.glob(str(Path(exe_config['tmp_folder']) / f"{name}_*")))
                if len(existing) > 0:
                    uuids = existing[0].split('_')[-1]
                    logger.info(f'Reusing existing uuid={uuids}')
            else:
                uuids = uuid.uuid1()
            tmp_data_folder = str(Path(exe_config['tmp_folder']) / f"{name}_{uuids}")

        else:
            tmp_data_folder = str(Path(exe_config['tmp_folder']) / f"{name}")

        os.makedirs(tmp_data_folder, exist_ok=True)
        logger.debug(f'tmp_data_folder: {tmp_data_folder}')

        try:
            self.pdbDataHandler = PdbDataHandler(src_data_folder=src_data_folder, name=name,
                                                 pregrid_transformer_config=pregrid_transformer_config,
                                                 tmp_data_folder=tmp_data_folder,
                                                 pdb2pqrPath=exe_config['pdb2pqrPath'],
                                                 cleanup=exe_config.get('cleanup', False),
                                                 reuse_grids=reuse_grids
                                                 )

            raws, labels = self.pdbDataHandler.getRawsLabels(features=features, grid_config=grid_config)
            allowRotations = self.pdbDataHandler.checkRotations()

        except Exception as e:
            raise type(e)(f"Tmp folder: {tmp_data_folder}") from e

        self.ndim = raws.ndim
        self.shape = raws.shape

        raws = [raws]
        if phase == 'test':
            labels = None
        else:
            labels = [labels]
        weight_maps = None

        super().__init__(raws, labels, weight_maps,
                         featuresTypes=features.feature_types,
                         tmp_data_folder=tmp_data_folder,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         mirror_padding=mirror_padding,
                         instance_ratio=instance_ratio,
                         random_seed=random_seed,
                         allowRotations=allowRotations,
                         debug_str=self.name)

    def getStructure(self):
        return self.pdbDataHandler.getStructureLigand()[0]

    def __getitem__(self, idx):
        logger.debug(f'Getting idx {idx} from {self.name}')
        return self.name, self.pdbDataHandler, super(StandardPDBDataset, self).__getitem__(idx)

    @classmethod
    def collate_fn(cls, xs):
        return [x[0] for x in xs], [x[1] for x in xs], default_collate([x[2] for x in xs])

    @classmethod
    def prediction_collate(cls, batch):
        names = [name for name,_,_ in batch]
        pdbObjs = [x for _, x, _ in batch]
        assert all(y == names[0] for y in names)
        samples = [data for _,_,data in batch]
        return default_prediction_collate(samples)


    @classmethod
    def create_datasets(cls, dataset_config, features_config, phase):
        phase_config = dataset_config[phase]

        logger.info(f"Slice builder config: {phase_config['slice_builder']}")

        file_paths = phase_config['file_paths']
        file_paths = PdbDataHandler.traverse_pdb_paths(file_paths)

        args = [(file_path, name, dataset_config, phase, features_config) for file_path, name in file_paths]

        pdb_workers = dataset_config.get('pdb_workers', 0)

        if pdb_workers > 0:
            logger.info(f'Parallelizing dataset creation among {pdb_workers} workers')
            pool = Pool(processes=pdb_workers)
            return [x for x in pool.map(create_dataset, args) if x is not None]
        else:
            return [x for x in (create_dataset(arg) for arg in args) if x is not None]

def create_dataset(arg):
    file_path, name, dataset_config, phase, feature_config = arg
    phase_config = dataset_config[phase]

    features : BaseFeatureList = get_features(feature_config)

    # load data augmentation configuration
    transformer_config = phase_config['transformer']
    pregrid_transformer_config = phase_config.get('pdb_transformer', [])
    grid_config = dataset_config.get('grid_config', {})

    # load slice builder config
    slice_builder_config = phase_config['slice_builder']

    # load instance sampling configuration
    instance_ratio = phase_config.get('instance_ratio', None)
    random_seed = phase_config.get('random_seed', 0)

    exe_config = {k: dataset_config[k] for k in ['tmp_folder', 'pdb2pqrPath', 'cleanup', 'reuse_grids'] if k in dataset_config.keys()}

    try:
        logger.info(f'Loading {phase} set from: {file_path} named {name} ...')
        dataset = StandardPDBDataset(src_data_folder=file_path,
                                     name=name,
                                     exe_config=exe_config,
                                     phase=phase,
                                     slice_builder_config=slice_builder_config,
                                     features=features,
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


