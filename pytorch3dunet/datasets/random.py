import os
from pathlib import Path
from multiprocessing import Lock, Pool
from pytorch3dunet.augment.transforms import Phase
from pytorch3dunet.datasets.base import AbstractDataset
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.datasets.featurizer import get_features, ComposedFeatures
from typing import Iterable, Optional, Mapping
from pytorch3dunet.datasets.config import LoadersConfig, RandomDataConfig
import numpy as np
from torch.utils.data._utils.collate import default_collate

logger = get_logger('PdbDataset')
lock = Lock()

class RandomDataset(AbstractDataset):

    def __init__(self, name:str, tmp_folder:str,
                 phase:str,
                 grid_size:int,
                 features: ComposedFeatures,
                 transformer_config:Mapping):

        phase = Phase.from_str(phase)
        tmp_data_folder = str(Path(tmp_folder) / f"{name}")
        os.makedirs(tmp_data_folder, exist_ok=True)
        logger.debug(f'tmp_data_folder: {tmp_data_folder}')

        raws = np.random.normal(size=(features.num_features, grid_size, grid_size, grid_size))
        labels = np.random.choice([0,1], size=(grid_size, grid_size, grid_size))

        allowRotations = True
        self.ndim = raws.ndim
        self.shape = raws.shape

        if phase == 'test':
            labels = None

        super().__init__(name=name, raws=raws, labels=labels,
                         featuresTypes=features.feature_types,
                         tmp_data_folder=tmp_data_folder,
                         phase=phase,
                         transformer_config=transformer_config,
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
    def create_datasets(cls, loaders_config:LoadersConfig, pdb_workers:int, features_config, transformer_config, phase) -> Iterable[AbstractDataset]:
        random_data_config : RandomDataConfig = loaders_config.data_config

        names = [str(i) for i in range(random_data_config.num[Phase.from_str(phase)])]

        args = ((name, loaders_config, phase, features_config, transformer_config) for name in names)

        if pdb_workers > 0:
            logger.info(f'Parallelizing dataset creation among {pdb_workers} workers')
            pool = Pool(processes=pdb_workers)
            return [x for x in pool.map(create_dataset, args) if x is not None]
        else:
            return [x for x in (create_dataset(arg) for arg in args) if x is not None]


def create_dataset(arg) -> Optional[RandomDataset]:
    name, loaders_config, phase, feature_config, transformer_config = arg

    features: ComposedFeatures = get_features(feature_config)

    grid_size = loaders_config.grid_size

    logger.info(f'Creating {phase} set named {name} ...')
    dataset = RandomDataset(name=name,
                            phase=phase,
                            features=features,
                            transformer_config=transformer_config,
                            tmp_folder=loaders_config.tmp_folder,
                            grid_size=grid_size)
    return dataset


