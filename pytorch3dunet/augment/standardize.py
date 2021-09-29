from pytorch3dunet.augment.transforms import TransformOptions, LocalTransform, Phase, \
    SkippableTransformOptions, SkippedTransform, logger, MyGenerator
from dataclasses import dataclass
from pytorch3dunet.datasets.featurizer import Transformable, PotentialGrid
from typing import Type, Mapping, Iterable, Any, Callable
import numpy as np

class Stats:
    mean: np.ndarray
    std: np.ndarray

    def __init__(self, raws:np.ndarray):
        assert raws.ndim == 4
        self.mean = raws.mean(axis=(1,2,3))
        self.std = raws.std(axis=(1, 2, 3))
        logger.debug(f'mean={self.mean}, std={self.std}')

@dataclass(frozen=True)
class StandardizeGlobalOptions(TransformOptions):
    eps: float = 1e-10

    def __post_init__(self):
        assert isinstance(self.eps, float)


@dataclass(frozen=True)
class StandardizeLocalOptions(TransformOptions):
    # TODO add option to use global stats
    pass


class Standardize(LocalTransform):
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    """

    @classmethod
    def is_rotation(cls):
        return False

    @classmethod
    def local_option_type(cls) -> Type[TransformOptions]:
        return StandardizeLocalOptions

    @classmethod
    def default_global_options(cls, phase: Phase) -> SkippableTransformOptions:
        return StandardizeGlobalOptions()

    @classmethod
    def default_local_options(cls, phase: Phase, ft: Type[Transformable]) -> SkippableTransformOptions:
        if ft == PotentialGrid:
            return StandardizeLocalOptions()
        return SkippedTransform()

    @classmethod
    def global_option_type(cls) -> Type[TransformOptions]:
        return StandardizeGlobalOptions


    def __init__(self, options_conf: Mapping[str, Mapping[str, Any]], phase: Phase, generator:MyGenerator,
                 featureStats: Stats, **kwargs):
        assert isinstance(featureStats, Stats)
        self.featureStats = featureStats
        super().__init__(options_conf, phase, generator)


    def makeCallableSequence(self, global_opt:StandardizeGlobalOptions) -> Iterable[Callable[[np.ndarray,TransformOptions,int], np.ndarray]]:

        # TODO assert that stats.shape == len(features)

        def func(m3d: np.ndarray, option: StandardizeLocalOptions, idx: int) -> np.ndarray:
            mean,std = self.featureStats.mean[idx],  self.featureStats.std[idx]
            return (m3d - mean) / np.clip(std, a_min=global_opt.eps, a_max=None)
        return [func]


