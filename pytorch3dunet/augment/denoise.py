from pytorch3dunet.augment.transforms import TransformOptions, LocalTransform, Phase, \
    SkippableTransformOptions, SkippedTransform, logger, MyGenerator
from dataclasses import dataclass
from pytorch3dunet.datasets.featurizer import Transformable
from pytorch3dunet.datasets.features import PotentialGrid
from typing import Type, Mapping, Iterable, Any, Callable, Optional
import numpy as np
from skimage.restoration import denoise_wavelet


@dataclass(frozen=True)
class DenoiseGlobalOptions(TransformOptions):
    pass


@dataclass(frozen=True)
class DenoiseLocalOptions(TransformOptions):
    wavelet: str = 'db1'
    mode: str = 'soft'
    wavelet_levels: Optional[int] = None
    sigma: Optional[float] = None
    method: str = 'BayesShrink'
    rescale_sigma: bool = True
    pass


class Denoise(LocalTransform):

    @classmethod
    def is_rotation(cls):
        return False

    @classmethod
    def is_random(cls):
        return False

    @classmethod
    def local_option_type(cls) -> Type[TransformOptions]:
        return DenoiseLocalOptions

    @classmethod
    def default_global_options(cls, phase: Phase) -> SkippableTransformOptions:
        return DenoiseGlobalOptions()

    @classmethod
    def default_local_options(cls, phase: Phase, ft: Type[Transformable]) -> SkippableTransformOptions:
        if ft == PotentialGrid:
            return DenoiseLocalOptions()
        return SkippedTransform()

    @classmethod
    def global_option_type(cls) -> Type[TransformOptions]:
        return DenoiseGlobalOptions

    def __init__(self, options_conf: Mapping[str, Mapping[str, Any]], phase: Phase, generator: MyGenerator, **kwargs):
        super().__init__(options_conf, phase, generator)

    def makeCallableSequence(self, global_opt: DenoiseGlobalOptions) -> Iterable[Callable[[np.ndarray, TransformOptions, int], np.ndarray]]:

        def func(m3d: np.ndarray, option: DenoiseLocalOptions, idx: int) -> np.ndarray:
            return denoise_wavelet(m3d, sigma=option.sigma, wavelet=option.wavelet, mode=option.mode,
                                   wavelet_levels=option.wavelet_levels, method=option.method, rescale_sigma=option.rescale_sigma)

        yield func
