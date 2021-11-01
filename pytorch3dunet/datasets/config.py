from __future__ import annotations
from pytorch3dunet.unet3d.utils import get_logger, Phase
from typing import Mapping, Union, List
from pathlib import Path
import itertools
import pprint

default_grid_size = 161
default_ligand_mask_radius = 6.5
default_batch_size = 1
default_pin_memory = False
default_benchmark = False
default_mixed = False

default_pdb2pqrPath = 'pdb2pqr'

logger = get_logger('Loaders')


def todict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, todict(value, classkey))
            for key, value in obj.__dict__.items()
            if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj


def default_if_none(x, default):
    return x if x is not None else default


class PdbDataConfig:
    data_paths = Mapping[Phase, List[str]]
    pdb2pqrPath: str = default_pdb2pqrPath
    reuse_grids: bool = False
    randomize_name: bool = False
    ligand_mask_radius: float = default_ligand_mask_radius

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

    def pretty_format(self):
        out = {k: v for k, v in vars(self).items() if k != 'data_paths'}
        out['data_paths'] = vars(self.data_paths)
        return pprint.pformat(out, indent=4)


class RandomDataConfig:
    num: Mapping[Phase,int]

    def __init__(self, train: int, val: int, test: int):
        self.num = {Phase.TRAIN: train, Phase.TEST: test, Phase.VAL: val}


class LoadersConfig:
    dataset_cls_str: str
    data_config = Union[PdbDataConfig, RandomDataConfig]
    num_workers: int
    batch_size: int = default_batch_size
    force_rotations: bool = False
    fail_on_error: bool = False
    random_mode: bool = False
    cleanup: bool = False
    pin_memory: bool = default_pin_memory
    grid_size: int = default_grid_size

    def __init__(self, runFolder: Path, runconfig: Mapping,
                 nworkers: int,
                 grid_size: Mapping,
                 dataset: str = None,
                 batch_size: int = None,
                 fail_on_error: bool = None,
                 force_rotations: bool = None,
                 cleanup: bool = None,
                 pin_memory: bool = None):

        runconfig = dict(runconfig)

        self.num_workers = nworkers

        self.batch_size = default_if_none(batch_size, self.batch_size)
        self.grid_size = default_if_none(grid_size, self.grid_size)
        self.fail_on_error = default_if_none(fail_on_error, self.fail_on_error)
        self.force_rotations = default_if_none(force_rotations, self.force_rotations)
        self.cleanup = default_if_none(cleanup, self.cleanup)
        self.pin_memory = default_if_none(pin_memory, self.pin_memory)

        self.tmp_folder = str(runFolder / 'tmp')

        self.random_mode = runconfig.pop('random', self.random_mode)

        if self.random_mode:
            self.dataset_cls_str = 'RandomDataset' if dataset is None else dataset
            self.data_config = RandomDataConfig(**runconfig)
        else:
            self.dataset_cls_str = 'PDBDataset' if dataset is None else dataset
            self.data_config = PdbDataConfig(**runconfig)


class RunConfig:
    loaders_config: LoadersConfig
    pdb_workers: int
    profile: bool
    benchmark: bool = default_benchmark
    mixed: bool = default_mixed
    max_gpus: int = None

    def __init__(self, runFolder: Path, runconfig: Mapping,
                 nworkers: int, pdb_workers: int, max_gpus: int,
                 loaders_config: Mapping,
                 profile:bool):
        runconfig = dict(runconfig)
        self.pdb_workers = pdb_workers
        self.max_gpus = max_gpus
        self.benchmark = runconfig.pop('benchmark', self.benchmark)
        self.mixed = runconfig.pop('mixed', self.mixed)
        self.loaders_config = LoadersConfig(runFolder, runconfig, nworkers, **loaders_config)
        self.profile = profile

    def pretty_format(self):
        return pprint.pformat(todict(self), indent=4)
