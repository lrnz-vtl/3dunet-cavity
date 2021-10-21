import os
from pathlib import Path
from multiprocessing import Lock, Pool
import h5py
import numpy as np
import glob
from pytorch3dunet.augment.utils import Transformer
from pytorch3dunet.augment.transforms import Transform, Phase
from pytorch3dunet.augment.standardize import Stats
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset, sample_instances, \
    default_prediction_collate
from pytorch3dunet.datasets.utils_pdb import PdbDataHandler
from pytorch3dunet.unet3d.utils import get_logger
import uuid
from torch.utils.data._utils.collate import default_collate
from pytorch3dunet.datasets.featurizer import BaseFeatureList, get_features, Transformable, LabelClass, ComposedFeatures
from typing import Iterable, Type, List, Optional
from abc import ABC

logger = get_logger('PdbDataset')
lock = Lock()


class AbstractDataset(ConfigDataset, ABC):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def set_transform_seeds(self, seed:int) -> None:
        self.raw_transform.set_seeds(seed)
        self.label_transform.set_seeds(seed)

    def __init__(self, raws:np.ndarray, labels:np.ndarray,
                 featuresTypes: Iterable[Type[Transformable]],
                 tmp_data_folder,
                 phase: Phase,
                 slice_builder_config,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
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
        self.featureTypes: Iterable[Type[Transformable]] = featuresTypes

        labels = np.expand_dims(labels, axis=0)

        if phase in [Phase.TRAIN, phase.VAL]:
            mirror_padding = None

        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"

        self.mirror_padding = mirror_padding
        self.phase = phase

        self.stats = Stats(raws)

        self.transformer = Transformer(transformer_config=transformer_config, common_config={},
                                       allowRotations=allowRotations, debug_str=debug_str, stats=self.stats)
        self.raw_transform = self.transformer.create_transform(self.phase, '_raw')

        if phase != Phase.TEST:
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.create_transform(self.phase, '_label')

            self._check_dimensionality(raws, labels)
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            labels = None

            # add mirror padding if needed
            if self.mirror_padding is not None:
                raise NotImplementedError('mirror_padding branch of the code has not been tested')
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                channels = [np.pad(r, pad_width=pad_width, mode='reflect') for r in raw]
                padded_volume = np.stack(channels)

                raws = padded_volume

        # build slice indices for raw and label data sets
        slice_builder = get_slice_builder(raws, labels, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices

        with h5py.File(self.h5path, 'w') as h5:
            h5.create_dataset('raws', data=raws)
            if labels is not None:
                h5.create_dataset('labels', data=labels)

        self.patch_count = len(self.raw_slices)
        logger.debug(f'Number of patches: {self.patch_count}')

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        with h5py.File(self.h5path, 'r') as h5:
            raws = np.array(h5['raws'])

            # get the slice for a given index 'idx'
            raw_idx = self.raw_slices[idx]
            # get the raw data patch for a given slice
            raw_patch_transformed = self.raw_transform(raws[raw_idx], self.featureTypes)

            if self.phase == Phase.TEST:
                # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
                if len(raw_idx) == 4:
                    raw_idx = raw_idx[1:]
                return (raw_patch_transformed, raw_idx)
            else:
                labels = np.array(h5['labels']).astype("<f8")
                # get the slice for a given index 'idx'
                label_idx = self.label_slices[idx]
                label_patch_transformed = self.label_transform(labels[label_idx], [LabelClass])

                # return the transformed raw and label patches
                return (raw_patch_transformed, label_patch_transformed)

    def _transform_patches(self, dataset, label_idx, transformer:Transform):
        transformed_patch = transformer(dataset[label_idx], self.featureTypes)
        return transformed_patch

    def __len__(self):
        return self.patch_count

    @staticmethod
    def _check_dimensionality(raws, labels):
        assert raws.ndim == 4
        assert labels.ndim == 4

        assert raws.shape[1:] == labels.shape[1:]


class StandardPDBDataset(AbstractDataset):

    def __init__(self, src_data_folder, name, exe_config,
                 phase:str,
                 slice_builder_config,
                 pregrid_transformer_config,
                 grid_config,
                 features: ComposedFeatures,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
                 random_seed=0,
                 force_rotations=False):

        reuse_grids = exe_config.get('reuse_grids', False)
        randomize_name = exe_config.get('randomize_name', False)

        self.src_data_folder = src_data_folder
        self.name = name

        phase = Phase.from_str(phase)

        if randomize_name:
            uuids = None
            if reuse_grids:
                try:
                    existing = next(glob.iglob(str(Path(exe_config['tmp_folder']) / f"{name}_*")))
                    uuids = existing.split('_')[-1]
                    logger.info(f'Reusing existing uuid={uuids}')
                except StopIteration:
                    pass
            if uuids is None:
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
            if force_rotations:
                allowRotations = True
                logger.warn('Forcing rotations for debugging')

        except Exception as e:
            raise type(e)(f"Tmp folder: {tmp_data_folder}") from e

        self.ndim = raws.ndim
        self.shape = raws.shape

        if phase == 'test':
            labels = None

        super().__init__(raws, labels,
                         featuresTypes=features.feature_types,
                         tmp_data_folder=tmp_data_folder,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         mirror_padding=mirror_padding,
                         random_seed=random_seed,
                         allowRotations=allowRotations,
                         debug_str=f'{self.name}')

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
    def create_datasets(cls, dataset_config, features_config, transformer_config, phase) -> Iterable[ConfigDataset]:
        phase_config = dataset_config[phase]

        logger.info(f"Slice builder config: {phase_config['slice_builder']}")

        file_paths = phase_config['file_paths']
        file_paths = PdbDataHandler.traverse_pdb_paths(file_paths)

        args = ((file_path, name, dataset_config, phase, features_config, transformer_config) for file_path, name in file_paths)

        pdb_workers = dataset_config.get('pdb_workers', 0)

        if pdb_workers > 0:
            logger.info(f'Parallelizing dataset creation among {pdb_workers} workers')
            pool = Pool(processes=pdb_workers)
            return [x for x in pool.map(create_dataset, args) if x is not None]
        else:
            return [x for x in (create_dataset(arg) for arg in args) if x is not None]


def create_dataset(arg) -> Optional[StandardPDBDataset]:
    file_path, name, dataset_config, phase, feature_config, transformer_config = arg
    phase_config = dataset_config[phase]
    fail_on_error = dataset_config.get('fail_on_error', False)
    force_rotations = dataset_config.get('force_rotations', False)

    features: ComposedFeatures = get_features(feature_config)

    # load data augmentation configuration
    pregrid_transformer_config = phase_config.get('pdb_transformer', [])
    grid_config = dataset_config.get('grid_config', {})

    # load slice builder config
    slice_builder_config = phase_config['slice_builder']

    # load instance sampling configuration
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
                                     random_seed=random_seed,
                                     force_rotations=force_rotations)
        return dataset
    except Exception as e:
        if fail_on_error:
            raise e
        logger.error(f'Skipping {phase} set from: {file_path} named {name}.', exc_info=True)
        return None


