from pytorch3dunet.augment.transforms import TransformOptions, LocalTransform, Phase, \
    SkippableTransformOptions, SkippedTransform, logger, MyGenerator
from dataclasses import dataclass
from pytorch3dunet.datasets.featurizer import Transformable, LabelClass
from pytorch3dunet.datasets.features import PotentialGrid
from typing import Type, Mapping, Iterable, Any, Callable, Optional
import numpy as np
from skimage.transform import rescale


@dataclass(frozen=True)
class DownscaleGlobalOptions(TransformOptions):
    scale: float


@dataclass(frozen=True)
class DownscaleLocalOptions(TransformOptions):
    order: int
    anti_aliasing: bool


class Downscale(LocalTransform):

    @classmethod
    def is_rotation(cls):
        return False

    @classmethod
    def is_random(cls):
        return False

    @classmethod
    def local_option_type(cls) -> Type[TransformOptions]:
        return DownscaleLocalOptions

    @classmethod
    def default_global_options(cls, phase: Phase) -> SkippableTransformOptions:
        # This will resize from 161^3 to 100^3
        return DownscaleGlobalOptions(0.62)

    @classmethod
    def default_local_options(cls, phase: Phase, ft: Type[Transformable]) -> SkippableTransformOptions:
        if ft == LabelClass:
            return DownscaleLocalOptions(anti_aliasing=False, order=0)
        else:
            return DownscaleLocalOptions(anti_aliasing=True, order=2)

    @classmethod
    def global_option_type(cls) -> Type[TransformOptions]:
        return DownscaleGlobalOptions

    def __init__(self, options_conf: Mapping[str, Mapping[str, Any]], phase: Phase, generator: MyGenerator, **kwargs):
        super().__init__(options_conf, phase, generator)

    def makeCallableSequence(self, global_opt: DownscaleGlobalOptions) -> Iterable[Callable[[np.ndarray, TransformOptions, int], np.ndarray]]:

        def func(m3d: np.ndarray, option: DownscaleLocalOptions, idx: int) -> np.ndarray:
            return rescale(m3d, scale=global_opt.scale, anti_aliasing=option.anti_aliasing, order=option.order)

        yield func
