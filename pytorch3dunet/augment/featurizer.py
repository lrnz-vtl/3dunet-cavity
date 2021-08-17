import importlib

import numpy as np

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
GLOBAL_RANDOM_STATE = np.random.RandomState(47)


class PotentialGrid():
    def __init__(self, **kwargs):
        pass

    def __call__(self, structure, grid):
        return grid.grid

class AtomLabel():

    def __init__(self, **kwargs):
        pass

    def __call__(self, structure, grid):
        retgrid = np.zeros(shape=grid.grid.shape)

        for i, coord in enumerate(structure.getCoords()):
            x, y, z = coord
            binx = int((x - min(grid.edges[0])) / grid.delta[0])
            biny = int((y - min(grid.edges[1])) / grid.delta[1])
            binz = int((z - min(grid.edges[2])) / grid.delta[2])

            retgrid[binx, biny, binz] = 1

        return retgrid

def get_featurizer(config):
    return Featurizer(config)

def ComposeFeatures(fts):
    def fullFt(*args):
        if len(fts) == 1:
            return fts[0](*args)
        return np.stack([ft(*args) for ft in fts])
    return fullFt

class Featurizer:
    def __init__(self, config):
        self.config = config
        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)

    def raw_transform(self):
        return self._create_featurizer()

    @staticmethod
    def _featurizer_class(class_name):
        m = importlib.import_module('pytorch3dunet.augment.featurizer')
        clazz = getattr(m, class_name)
        return clazz

    def _create_featurizer(self):
        return ComposeFeatures([
            self._create_feature(c) for c in self.config
        ])

    def _create_feature(self, c):
        config = c
        config['random_state'] = np.random.RandomState(self.seed)
        ft_class = self._featurizer_class(config['name'])
        return ft_class(**config)