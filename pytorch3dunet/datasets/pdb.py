import os
from pathlib import Path
from multiprocessing import Lock, Pool
import glob
from pytorch3dunet.augment.transforms import Phase
from pytorch3dunet.datasets.base import AbstractDataset, default_prediction_collate
from pytorch3dunet.datasets.config import LoadersConfig, PdbDataConfig
from pytorch3dunet.datasets.utils_pdb import PdbDataHandler
from pytorch3dunet.unet3d.utils import get_logger
import uuid
from torch.utils.data._utils.collate import default_collate
from pytorch3dunet.datasets.featurizer import get_features, ComposedFeatures
from typing import Iterable, Optional
from dataclasses import  dataclass

logger = get_logger('PdbDataset')
lock = Lock()

@dataclass
class ExeConfig:
    tmp_folder : str
    pdb2pqrPath: str
    cleanup: bool
    reuse_grids : bool
    randomize_name: bool

class PDBDataset(AbstractDataset):

    def __init__(self, src_data_folder, name, exe_config : ExeConfig,
                 phase:str,
                 grid_size: int,
                 ligand_mask_radius:float,
                 features: ComposedFeatures,
                 transformer_config,
                 force_rotations=False):

        reuse_grids = exe_config.reuse_grids
        randomize_name = exe_config.randomize_name

        self.src_data_folder = src_data_folder

        phase = Phase.from_str(phase)

        if randomize_name:
            uuids = None
            if reuse_grids:
                try:
                    existing = next(glob.iglob(str(Path(exe_config.tmp_folder) / f"{name}_*")))
                    uuids = existing.split('_')[-1]
                    logger.info(f'Reusing existing uuid={uuids}')
                except StopIteration:
                    pass
            if uuids is None:
                uuids = uuid.uuid1()
            tmp_data_folder = str(Path(exe_config.tmp_folder) / f"{name}_{uuids}")

        else:
            tmp_data_folder = str(Path(exe_config.tmp_folder) / f"{name}")

        os.makedirs(tmp_data_folder, exist_ok=True)
        logger.debug(f'tmp_data_folder: {tmp_data_folder}')

        try:
            self.pdbDataHandler = PdbDataHandler(src_data_folder=src_data_folder, name=name,
                                                 tmp_data_folder=tmp_data_folder,
                                                 pdb2pqrPath=exe_config.pdb2pqrPath,
                                                 cleanup=exe_config.cleanup,
                                                 reuse_grids=reuse_grids,
                                                 )

            raws, labels = self.pdbDataHandler.getRawsLabels(features=features, grid_size=grid_size, ligand_mask_radius=ligand_mask_radius)
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

        super().__init__(name=name, raws=raws, labels=labels,
                         featuresTypes=features.feature_types,
                         tmp_data_folder=tmp_data_folder,
                         phase=phase,
                         transformer_config=transformer_config,
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
    def create_datasets(cls, loaders_config: LoadersConfig, features_config, transformer_config, phase) -> Iterable[AbstractDataset]:

        data_paths = loaders_config.data_config.data_paths
        file_paths = data_paths[Phase.from_str(phase)]
        file_paths = PdbDataHandler.traverse_pdb_paths(file_paths)

        args = ((file_path, name, loaders_config, phase, features_config, transformer_config) for file_path, name in file_paths)

        pdb_workers = loaders_config.pdb_workers

        if pdb_workers > 0:
            logger.info(f'Parallelizing dataset creation among {pdb_workers} workers')
            pool = Pool(processes=pdb_workers)
            return [x for x in pool.map(create_dataset, args) if x is not None]
        else:
            return [x for x in (create_dataset(arg) for arg in args) if x is not None]




def create_dataset(arg) -> Optional[PDBDataset]:
    file_path, name, loaders_config, phase, feature_config, transformer_config = arg
    fail_on_error = loaders_config.fail_on_error
    force_rotations = loaders_config.force_rotations

    features: ComposedFeatures = get_features(feature_config)

    pdb_config : PdbDataConfig = loaders_config.data_config

    grid_size = loaders_config.grid_size

    exe_config = ExeConfig(tmp_folder=loaders_config.tmp_folder,
                           pdb2pqrPath=pdb_config.pdb2pqrPath,
                           cleanup=loaders_config.cleanup,
                           reuse_grids=pdb_config.reuse_grids,
                           randomize_name=pdb_config.randomize_name)

    try:
        logger.info(f'Loading {phase} set from: {file_path} named {name} ...')
        dataset = PDBDataset(src_data_folder=file_path,
                             name=name,
                             exe_config=exe_config,
                             phase=phase,
                             features=features,
                             transformer_config=transformer_config,
                             grid_size=grid_size,
                             force_rotations=force_rotations,
                             ligand_mask_radius=pdb_config.ligand_mask_radius)
        return dataset
    except Exception as e:
        if fail_on_error:
            raise e
        logger.error(f'Skipping {phase} set from: {file_path} named {name}.', exc_info=True)
        return None


