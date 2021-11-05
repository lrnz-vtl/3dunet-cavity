import numpy as np
import itertools
from typing import List, Type
from pytorch3dunet.unet3d.utils import get_logger, get_attr
from abc import ABC, abstractmethod

logger = get_logger('Featurizer')

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
GLOBAL_RANDOM_STATE = np.random.RandomState(47)

try:
    from prody import AtomGroup
except ImportError:
    AtomGroup = type('AtomGroup', (), {})
try:
    from openbabel.pybel import Molecule
except ImportError:
    Molecule = type('Molecule', (), {})
try:
    from pytorch3dunet.datasets.apbs import ApbsGridCollection
except ImportError:
    ApbsGridCollection = type('ApbsGridCollection', (), {})

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
    def call(self, structure:AtomGroup, mol:Molecule, grids: ApbsGridCollection) -> List[np.array]:
        raise NotImplementedError

    def __call__(self, *args) -> List[np.array]:
        ret = self.call(*args)
        assert len(ret) == self.num_features
        return ret

    def getDielecConstList(self):
        return []

class DummyFeature(BaseFeatureList):

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
        pass

    def getDielecConstList(self):
        return []

class ComposedFeatures(BaseFeatureList):

    @property
    def feature_types(self) -> List[Type[Transformable]]:
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
    modules = ['pytorch3dunet.datasets.featurizer']
    return get_attr(name, modules)


def get_features(configs) -> ComposedFeatures:
    modules = ['pytorch3dunet.datasets.featurizer', 'pytorch3dunet.datasets.features']

    def _create_feature(config):
        ft_class = get_attr(config['name'], modules)
        return ft_class(**config)

    return ComposedFeatures([_create_feature(config) for config in configs])