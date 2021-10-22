from __future__ import annotations
from pytorch3dunet.unet3d.utils import get_logger, Phase
from typing import Mapping, Union, List
from pathlib import Path
import itertools

MAX_SEED = 2 ** 32 - 1

logger = get_logger('Loaders')


def default_if_none(x, default):
    return x if x is not None else default


class PdbDataConfig:
    data_paths = Mapping[Phase, List[str]]
    pdb2pqrPath: str = 'pdb2pqr'
    reuse_grids: bool = False
    randomize_name: bool = False
    ligand_mask_radius: float = 6.5

    def __init__(self, dataFolder: str, train: List[str], val: List[str], test: List[str],
                 pdb2pqrPath: str = None, reuse_grids: bool = None, randomize_name: bool = None,
                 ligand_mask_radius: float = None):
        dataFolder = Path(dataFolder)

        self.data_paths = {Phase.TRAIN: [str(dataFolder / name) for name in train],
                           Phase.TEST: [str(dataFolder / name) for name in test],
                           Phase.VAL: [str(dataFolder / name) for name in val]}

        self.pdb2pqrPath = default_if_none(pdb2pqrPath, self.pdb2pqrPath)
        self.reuse_grids = default_if_none(reuse_grids, self.reuse_grids)
        self.randomize_name = default_if_none(randomize_name, self.randomize_name)
        self.ligand_mask_radius = default_if_none(ligand_mask_radius, self.ligand_mask_radius)

        for leftPhase, rightPhase in itertools.combinations(list(Phase), 2):
            left = self.data_paths[leftPhase]
            right = self.data_paths[rightPhase]
            assert set(left).isdisjoint(right), \
                f"{leftPhase} and {rightPhase} file paths overlap. Train, val and test sets must be separate"


class RandomDataConfig:
    num: Mapping[Phase,int]

    def __init__(self, train: int, val: int, test: int):
        self.num = {Phase.TRAIN: train, Phase.TEST: test, Phase.VAL: val}


class LoadersConfig:
    dataset_cls_str: str
    data_config = Union[PdbDataConfig, RandomDataConfig]
    batch_size: int = 1
    force_rotations: bool = False
    fail_on_error: bool = False
    random_mode: bool = False
    cleanup: bool = False
    grid_size: int = 161

    def __init__(self, runFolder: Path, runconfig: Mapping,
                 nworkers: int, pdbworkers: int,
                 grid_size: Mapping,
                 dataset: str = None,
                 batch_size: int = None,
                 fail_on_error: bool = None,
                 force_rotations: bool = None,
                 cleanup: bool = None):

        runconfig = dict(runconfig)

        self.num_workers = nworkers
        self.pdb_workers = pdbworkers
        self.batch_size = default_if_none(batch_size, self.batch_size)
        self.grid_size = default_if_none(grid_size, self.grid_size)
        self.fail_on_error = default_if_none(fail_on_error, self.fail_on_error)
        self.force_rotations = default_if_none(force_rotations, self.force_rotations)
        self.cleanup = default_if_none(cleanup, self.cleanup)

        self.tmp_folder = str(runFolder / 'tmp')

        self.random_mode = runconfig.get('random', self.random_mode)
        runconfig.pop('random', None)

        if self.random_mode:
            self.dataset_cls_str = 'RandomDataset' if dataset is None else dataset
            self.data_config = RandomDataConfig(**runconfig)
        else:
            self.dataset_cls_str = 'PDBDataset' if dataset is None else dataset
            self.data_config = PdbDataConfig(**runconfig)
