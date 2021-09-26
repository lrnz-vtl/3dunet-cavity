from pytorch3dunet.augment.transforms import TransformOptions, LocalTransform, Phase, \
    SkippableTransformOptions, SkippedTransform, PicklableGenerator, logger
from dataclasses import dataclass
from pytorch3dunet.datasets.featurizer import Transformable
from typing import Type, Mapping, Iterable, Any, Callable
import numpy as np
import torch
from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation
import numbers


@dataclass(frozen=True)
class RotateGlobalOptions(TransformOptions):
    prob: float = 1.0

    def __post_init__(self):
        assert isinstance(self.prob, numbers.Number)
        assert 1 >= self.prob >= 0


@dataclass(frozen=True)
class RotateLocalOptions(TransformOptions):
    mode: str
    cval: int = 0
    order: int = 3

    def __post_init__(self):
        assert isinstance(self.mode, str)
        assert isinstance(self.cval, int)
        assert isinstance(self.order, int)


class RandomRotate3D(LocalTransform):

    @classmethod
    def default_local_options(cls, phase: Phase, ft: Type[Transformable]) -> SkippableTransformOptions:
        return {
            'LabelClass': RotateLocalOptions(mode='constant'),
            'KalasantyFeatures': RotateLocalOptions(mode='constant'),
            'PotentialGrid': RotateLocalOptions(mode='nearest'),
            'AtomLabel': RotateLocalOptions(mode='constant'),
        }[ft.__name__]

    @classmethod
    def default_global_options(cls, phase: Phase) -> SkippableTransformOptions:
        if phase == phase.TRAIN:
            return RotateGlobalOptions()
        return SkippedTransform()

    @classmethod
    def global_option_type(cls) -> Type[TransformOptions]:
        return RotateGlobalOptions

    def __init__(self, options_conf: Mapping[str, Mapping[str, Any]], phase:Phase,
                 generator: PicklableGenerator,
                 **kwargs):
        self.generator = generator
        super().__init__(options_conf, phase)

    def makeCallableSequence(self, global_opt: RotateGlobalOptions) -> Iterable[Callable[[np.ndarray,TransformOptions,int], np.ndarray]]:

        rand = torch.rand(size=(1,), generator=self.generator).item()
        if rand > global_opt.prob:
            return []

        seed = self.generator.gen_seed()
        r = Rotation.random(random_state=seed)
        angles = r.as_euler('zxy')
        axes = [(0, 1), (1, 2), (0, 2)]

        for i, (axis, angle) in enumerate(zip(axes, angles)):
            angle = angle / np.pi * 180

            def func(m3d: np.array, opt: RotateLocalOptions, idx: int) -> np.array:
                return rotate(m3d, angle, axes=axis, reshape=False, order=opt.order, mode=opt.mode, cval=opt.cval)

            yield func
