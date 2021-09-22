import logging
from typing import Union, Callable, Iterable, Generator
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pytorch3dunet.augment.featurizer import BaseFeatureList
from typing import List, Type
import torch
import numpy as np
from pytorch3dunet.unet3d.utils import get_logger
# from collections.abc import Iterable

MAX_SEED = 2**32 - 1
GLOBAL_RANDOM_STATE = np.random.RandomState(47)

logger = get_logger('Transforms', level=logging.DEBUG)

class PicklableGenerator(torch.Generator):
    def __getstate__(self):
        return self.get_state()

    def __setstate__(self, state):
        return self.set_state(state)

    def gen_seed(self):
        return torch.randint(generator=self, high=MAX_SEED, size=(1,)).item()


class Phase(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class TransformOptions(ABC):
    pass
class SkippedTransform:
    pass
SkippableTransform = Union[TransformOptions, SkippedTransform]


class BaseTransform(ABC):

    def __init__(self, conf, phase: Phase, featureTypes: List[Type[BaseFeatureList]]):
        self._phase = phase
        self._conf = conf
        self._featureTypes = featureTypes

    @classmethod
    @abstractmethod
    def default_options(cls, phase: Phase, ft: Type[BaseFeatureList]) -> SkippableTransform:
        pass

    @abstractmethod
    def makeCallableSequence(self) -> Generator[Callable[[np.array,TransformOptions], np.array], None, None]:
        pass

    def __call__(self, m: np.array) -> np.array:
        # FIXME
        # assert m.ndim == 4
        assert m.shape[0] == len(self._featureTypes)
        shape = m.shape

        for callable in self.makeCallableSequence():
            channels = []
            for c, featureType in zip(range(m.shape[0]), self._featureTypes):
                opt = self._get_options(featureType)
                if isinstance(opt, SkippedTransform):
                    channels.append([c])
                else:
                    channels.append(callable(m[c], opt))
            m = np.stack(channels, axis=0)

        assert m.shape == shape
        return m

    def _get_options(self, cls: Type[BaseFeatureList]) -> SkippableTransform:
        if self._phase in self._conf.keys():
            opt = self._conf[self._phase].get(cls, self.default_options(self._phase, cls))
        else:
            opt =  self.default_options(self._phase, cls)
        logger.debug(f'cls = {cls}, option = {opt}')
        return opt


@dataclass(frozen=True)
class RandomFlipOptions(TransformOptions):
    axis_prob:int = 0.5


class RandomFlip(BaseTransform):
    """
    Randomly flips the image across the given axes.
    """

    @classmethod
    def default_options(cls, phase: Phase, ft: Type[BaseFeatureList]) -> SkippableTransform:
        if phase == Phase.TRAIN:
            return RandomFlipOptions(0.5)
        return SkippedTransform

    def __init__(self, conf, phase: Phase, featureTypes:List[Type[BaseFeatureList]],
                 generator: PicklableGenerator,
                 axes = (0, 1, 2),
                 **kwargs):
        self.random_state = np.random.RandomState(seed=generator.gen_seed())
        self.axes = axes
        super().__init__(conf, phase, featureTypes)

    def makeCallableSequence(self) -> Generator[Callable[[np.array,TransformOptions], np.array], None, None]:
        for axis in self.axes:
            rand = self.random_state.uniform()
            logger.debug(f'rand = {rand}')
            def func(m3d: np.array, option: RandomFlipOptions):
                if rand < option.axis_prob:
                    return np.flip(m3d, axis)
                else:
                    return m3d
            yield func