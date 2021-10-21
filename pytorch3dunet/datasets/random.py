import os
from pathlib import Path
from multiprocessing import Lock, Pool
from pytorch3dunet.augment.transforms import Phase
from pytorch3dunet.datasets.basedataset import AbstractDataset
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.datasets.featurizer import get_features, ComposedFeatures
from typing import Iterable, Optional
import numpy as np
from torch.utils.data._utils.collate import default_collate

logger = get_logger('PdbDataset')
lock = Lock()


class RandomDataset(AbstractDataset):

    def __init__(self, name, exe_config,
                 phase:str,
                 slice_builder_config,
                 grid_config,
                 features: ComposedFeatures,
                 transformer_config,
                 random_seed=0):

        phase = Phase.from_str(phase)
        tmp_data_folder = str(Path(exe_config['tmp_folder']) / f"{name}")
        os.makedirs(tmp_data_folder, exist_ok=True)
        logger.debug(f'tmp_data_folder: {tmp_data_folder}')

        grid_size = grid_config['grid_size']
        raws = np.random.normal(size=(features.num_features, grid_size, grid_size, grid_size))
        labels = np.random.choice([0,1], size=(grid_size, grid_size, grid_size))

        allowRotations = True
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
                         random_seed=random_seed,
                         allowRotations=allowRotations,
                         debug_str=f'{name}')


    @staticmethod
    def mock_traverse_paths(file_paths):

        assert isinstance(file_paths, list)
        results = []
        for file_path in file_paths:
            file_path = Path(file_path)
            name = str(file_path.name)
            results.append(name)
        return results

    @classmethod
    def collate_fn(cls, xs):
        return [x[0] for x in xs], [None for _ in xs], default_collate([x[1] for x in xs])

    def __getitem__(self, idx):
        logger.debug(f'Getting idx {idx} from {self.name}')
        return self.name, super().__getitem__(idx)

    @classmethod
    def create_datasets(cls, dataset_config, features_config, transformer_config, phase) -> Iterable[AbstractDataset]:
        phase_config = dataset_config[phase]

        logger.info(f"Slice builder config: {phase_config['slice_builder']}")

        file_paths = phase_config['file_paths']
        names = cls.mock_traverse_paths(file_paths)

        args = ((name, dataset_config, phase, features_config, transformer_config) for name in names)

        pdb_workers = dataset_config.get('pdb_workers', 0)

        if pdb_workers > 0:
            logger.info(f'Parallelizing dataset creation among {pdb_workers} workers')
            pool = Pool(processes=pdb_workers)
            return [x for x in pool.map(create_dataset, args) if x is not None]
        else:
            return [x for x in (create_dataset(arg) for arg in args) if x is not None]


def create_dataset(arg) -> Optional[RandomDataset]:
    name, dataset_config, phase, feature_config, transformer_config = arg
    phase_config = dataset_config[phase]

    features: ComposedFeatures = get_features(feature_config)

    # load data augmentation configuration
    grid_config = dataset_config.get('grid_config', {})

    # load slice builder config
    slice_builder_config = phase_config['slice_builder']

    # load instance sampling configuration
    random_seed = phase_config.get('random_seed', 0)

    exe_config = {k: dataset_config[k] for k in ['tmp_folder', 'pdb2pqrPath', 'cleanup', 'reuse_grids'] if k in dataset_config.keys()}

    logger.info(f'Creating {phase} set named {name} ...')
    dataset = RandomDataset(name=name,
                            exe_config=exe_config,
                            phase=phase,
                            slice_builder_config=slice_builder_config,
                            features=features,
                            transformer_config=transformer_config,
                            grid_config=grid_config,
                            random_seed=random_seed)
    return dataset


