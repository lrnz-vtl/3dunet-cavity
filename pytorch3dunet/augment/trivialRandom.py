from pytorch3dunet.augment.transforms import TransformOptions, LocalTransform, Phase, \
    SkippableTransformOptions, MyGenerator, logger
from dataclasses import dataclass
from pytorch3dunet.datasets.featurizer import Transformable
from typing import Type, Mapping, Iterable, Any, Callable
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import numbers


@dataclass(frozen=True)
class GlobalOptions(TransformOptions):
    prob: float = 1.0

    def __post_init__(self):
        assert isinstance(self.prob, numbers.Number)
        assert 1 >= self.prob >= 0


@dataclass(frozen=True)
class LocalOptions(TransformOptions):
    pass


class TrivialTransform(LocalTransform):

    @classmethod
    def local_option_type(cls) -> Type[TransformOptions]:
        return LocalOptions

    @classmethod
    def is_rotation(cls):
        return False

    @classmethod
    def default_local_options(cls, phase: Phase, ft: Type[Transformable]) -> SkippableTransformOptions:
        return {
            'LabelClass': LocalOptions(),
            'KalasantyFeatures': LocalOptions(),
            'PotentialGrid': LocalOptions(),
            'AtomLabel': LocalOptions(),
        }[ft.__name__]

    @classmethod
    def default_global_options(cls, phase: Phase) -> SkippableTransformOptions:
        return GlobalOptions()

    @classmethod
    def global_option_type(cls) -> Type[TransformOptions]:
        return GlobalOptions

    def __init__(self, options_conf: Mapping[str, Mapping[str, Any]], phase:Phase,
                 generator: MyGenerator, debug_str:str=None,
                 **kwargs):
        self.debug_str = debug_str
        super().__init__(options_conf, phase, generator)

    def makeCallableSequence(self, global_opt: GlobalOptions) -> Iterable[Callable[[np.ndarray,TransformOptions,int], np.ndarray]]:

        rand = torch.rand(size=(1,), generator=self.generator).item()
        if rand > global_opt.prob:
            return []

        seed = self.generator.gen_seed()
        r = Rotation.random(random_state=seed)

        return []