import os
from pathlib import Path
from multiprocessing import Lock, Pool
import glob
from pytorch3dunet.augment.transforms import Phase
from pytorch3dunet.datasets.basedataset import AbstractDataset, default_prediction_collate
from pytorch3dunet.datasets.utils_pdb import PdbDataHandler
from pytorch3dunet.unet3d.utils import get_logger
import uuid
from torch.utils.data._utils.collate import default_collate
from pytorch3dunet.datasets.featurizer import get_features, ComposedFeatures
from typing import Iterable, Optional

logger = get_logger('PdbDataset')
lock = Lock()


class PDBDataset(AbstractDataset):

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

        super().__init__(name, raws, labels,
                         featuresTypes=features.feature_types,
                         tmp_data_folder=tmp_data_folder,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         mirror_padding=mirror_padding,
                         random_seed=random_seed,
                         allowRotations=allowRotations,
                         debug_str=f'{name}')

    def getStructure(self):
        return self.pdbDataHandler.getStructureLigand()[0]

    def __getitem__(self, idx):
        logger.debug(f'Getting idx {idx} from {self.name}')
        return self.name, self.pdbDataHandler, super(PDBDataset, self).__getitem__(idx)

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
    def create_datasets(cls, dataset_config, features_config, transformer_config, phase) -> Iterable[AbstractDataset]:
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


def create_dataset(arg) -> Optional[PDBDataset]:
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
        dataset = PDBDataset(src_data_folder=file_path,
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


