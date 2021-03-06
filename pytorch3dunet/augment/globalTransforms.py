from pytorch3dunet.augment.transforms import TransformOptions, Phase, \
    SkippableTransformOptions, SkippedTransform, MyGenerator, BaseTransform, logger
from dataclasses import dataclass
from pytorch3dunet.datasets.featurizer import Transformable
from typing import Type, Mapping, List, Any
import numpy as np
import numbers
from pytorch3dunet.unet3d.utils import profile

MAX_SEED = 2**32 - 1
GLOBAL_RANDOM_STATE = np.random.RandomState(47)


@dataclass(frozen=True)
class RandomFlipOptions(TransformOptions):
    axis_prob: float = 0.5

    def __post_init__(self):
        assert isinstance(self.axis_prob, numbers.Number)
        assert 1 >= self.axis_prob >= 0


class RandomFlip(BaseTransform):
    """
    Randomly flips the image.
    """

    @classmethod
    def is_random(cls) -> bool:
        return True

    @classmethod
    def is_rotation(cls):
        return False

    @classmethod
    def global_option_type(cls) -> Type[TransformOptions]:
        return RandomFlipOptions

    @classmethod
    def default_global_options(cls, phase: Phase) -> SkippableTransformOptions:
        assert isinstance(phase, Phase)
        if phase == Phase.TRAIN:
            return RandomFlipOptions()
        return SkippedTransform()

    def __init__(self, options_conf: Mapping[str, Any], phase: Phase,
                 generator: MyGenerator, debug_str:str, **kwargs):
        self.debug_str = debug_str
        super().__init__(options_conf, phase, generator)

    @profile
    def _call(self, m: np.ndarray, global_opt: RandomFlipOptions, featureTypes: List[Type[Transformable]]) -> np.ndarray:
        axis = (1,)
        rand = np.random.RandomState(seed=self.generator.gen_seed()).uniform()
        logger.debug(f'Flip rand = {rand}, {self.debug_str}')
        if rand < global_opt.axis_prob:
            return np.flip(m, axis)
        return m