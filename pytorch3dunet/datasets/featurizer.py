import importlib
import numpy as np
import openbabel.pybel
import prody
import itertools
from typing import List, Type, Iterable, Mapping
import tfbio.data
from pytorch3dunet.unet3d.utils import get_logger
from potsim2 import PotGrid
from abc import ABC, abstractmethod

logger = get_logger('Featurizer')

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
GLOBAL_RANDOM_STATE = np.random.RandomState(47)


class ApbsGridCollection:
    """ Custom class representing a single electric potential grid """
    def __init__(self, pot_grids: Mapping[float, PotGrid], grid_size):
        """
        pot_grids: Mapping dielectric constant -> PotGrid
        """
        firstGrid: PotGrid = next(iter(pot_grids.values()))
        assert all(pot_grid.grid.shape == firstGrid.grid.shape for pot_grid in pot_grids.values())
        assert all((np.array(pot_grid.edges) == np.array(firstGrid.edges)).all() for pot_grid in pot_grids.values())
        assert all((pot_grid.delta == firstGrid.delta).all() for pot_grid in pot_grids.values())

        assert firstGrid.grid.shape[0] >= grid_size

        self.orig_shape = firstGrid.grid.shape
        self.grid_size = grid_size
        if self.grid_size < firstGrid.grid.shape[0]:
            logger.warn(
                f"Requested grid_size = {self.grid_size} is smaller than apbs output grid size = {firstGrid.grid.shape[0]}. "
                f"Applying naive cropping!!!")

        self.grids = {k: pot_grid.grid[:self.grid_size, :self.grid_size, :self.grid_size] for k,pot_grid in pot_grids.items()}
        self.edges = [x[:self.grid_size] for x in firstGrid.edges]
        self.delta = firstGrid.delta
        self.shape = next(iter(self.grids.values())).shape

    def homologate_labels(self, labels):
        assert labels.shape == self.orig_shape
        return labels[:self.grid_size, :self.grid_size, :self.grid_size]

    def delGrids(self):
        """ Free memory """
        del self.grids


class Transformable(ABC):

    @property
    @abstractmethod
    def feature_types(self) -> List[type]:
        pass

    @property
    @abstractmethod
    def num_features(self) -> int:
        pass

    @property
    @abstractmethod
    def names(self) -> List[str]:
        pass


class LabelClass(Transformable):

    num_features = 1

    @property
    def feature_types(self) -> List[type]:
        return [type(self)]

    @property
    def names(self):
        return [type(self).__name__]

    def __init__(self, **kwargs):
        pass

class BaseFeatureList(Transformable, ABC):

    @abstractmethod
    def call(self, structure:prody.AtomGroup, mol:openbabel.pybel.Molecule, grids: ApbsGridCollection) -> List[np.array]:
        raise NotImplementedError

    def __call__(self, *args) -> List[np.array]:
        ret = self.call(*args)
        assert len(ret) == self.num_features
        return ret

    def getDielecConstList(self):
        return []


class PotentialGrid(BaseFeatureList):

    num_features = 1
    dielec_const_default = 4.0

    @property
    def feature_types(self) -> List[type]:
        return [type(self)]

    @property
    def names(self):
        return [type(self).__name__]

    def __init__(self, **kwargs):
        self.dielec_const = kwargs.get('dielec_const', self.dielec_const_default)

    def call(self, structure, mol, grids):
        return [grids.grids[self.dielec_const]]

    def getDielecConstList(self):
        return [self.dielec_const]


class AtomLabel(BaseFeatureList):

    num_features = 1

    @property
    def feature_types(self) -> List[type]:
        return [type(self)]

    @property
    def names(self):
        return [type(self).__name__]

    def __init__(self, **kwargs):
        pass

    def call(self, structure, mol, grids):
        retgrid = np.zeros(shape=grids.shape)

        for i, coord in enumerate(structure.getCoords()):
            x, y, z = coord
            binx = int((x - min(grids.edges[0])) / grids.delta[0])
            biny = int((y - min(grids.edges[1])) / grids.delta[1])
            binz = int((z - min(grids.edges[2])) / grids.delta[2])

            retgrid[binx, biny, binz] = 1

        return [retgrid]


class KalasantyFeatures(BaseFeatureList):

    @property
    def feature_types(self) -> List[type]:
        return [type(self)] * self.num_features

    @property
    def names(self):
        return self._names

    @property
    def num_features(self):
        return len(self._names)

    def __init__(self, **kwargs):
        self.featurizer = tfbio.data.Featurizer(save_molecule_codes=False)
        self._names = self.featurizer.FEATURE_NAMES

    def call(self, structure, mol, grids):
        prot_coords, prot_features = self.featurizer.get_features(mol)
        max_dist = float(grids.grid_size - 1)/2.0
        box_center = np.array([(edge[0] + edge[-1]) / 2.0 for edge in grids.edges])
        # the coordinates must be relative to the box center
        prot_coords -= box_center
        ret = tfbio.data.make_grid(prot_coords, prot_features, max_dist=max_dist, grid_resolution=1.0)
        return list(ret)


class ComposedFeatures(BaseFeatureList):

    @property
    def feature_types(self) -> List[type]:
        return self._types

    @property
    def names(self):
        return self._names

    @property
    def num_features(self):
        return self._num_features

    def __init__(self, fts: List[BaseFeatureList]):
        self.fts = fts
        self._names = list(itertools.chain.from_iterable((ft.names for ft in fts)))
        self._types = list(itertools.chain.from_iterable(([type(ft)]*ft.num_features for ft in fts)))
        self._num_features = sum(ft.num_features for ft in fts)
        assert len(self.names) == self.num_features

    def call(self, *args):
        return np.stack(list(itertools.chain.from_iterable((ft(*args) for ft in self.fts))))

    def getDielecConstList(self):
        return list(itertools.chain.from_iterable((ft.getDielecConstList() for ft in self.fts)))


def get_feature_cls(name) -> Type[Transformable]:
    m = importlib.import_module('pytorch3dunet.datasets.featurizer')
    ft_class = getattr(m, name)
    return ft_class


def get_features(configs) -> ComposedFeatures:

    def _create_feature(config):
        m = importlib.import_module('pytorch3dunet.datasets.featurizer')
        ft_class = getattr(m, config['name'])
        return ft_class(**config)

    return ComposedFeatures([_create_feature(config) for config in configs])