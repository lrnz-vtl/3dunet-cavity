import importlib
import numpy as np
import openbabel.pybel
import prody
import itertools
from typing import List, Type
import tfbio.data
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.datasets.featurizer import Transformable, BaseFeatureList, ComposedFeatures

logger = get_logger('Featurizer')


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
        self.warned_small_grid = False

    def call(self, structure, mol, grids):
        retgrid = np.zeros(shape=grids.shape)

        for i, coord in enumerate(structure.getCoords()):
            x, y, z = coord
            binx = int((x - min(grids.edges[0])) / grids.delta[0])
            biny = int((y - min(grids.edges[1])) / grids.delta[1])
            binz = int((z - min(grids.edges[2])) / grids.delta[2])

            if binx < grids.shape[0] and biny < grids.shape[1] and binz < grids.shape[2]:
                retgrid[binx, biny, binz] = 1
            elif not self.warned_small_grid:
                logger.warn("Using small grid size so discarding data!")
                self.warned_small_grid = True

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