import dataclasses
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from pytorch3dunet.datasets.featurizer import Transformable, get_feature_cls
from typing import List, Type, Iterable, Any, Callable, Mapping, Union
import torch
import numpy as np
from pytorch3dunet.unet3d.utils import get_logger
import inspect
from pytorch3dunet.unet3d.utils import profile

MAX_SEED = 2 ** 32 - 1
GLOBAL_RANDOM_STATE = np.random.RandomState(47)

logger = get_logger('Transformer')


class MyGenerator(torch.Generator):
    def gen_seed(self):
        return torch.randint(generator=self, high=MAX_SEED, size=(1,)).item()


class Phase(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3

    @classmethod
    def from_str(cls, x:str):
        y = x.lower()
        if y=='train':
            return cls.TRAIN
        if y=='test':
            return cls.TEST
        if y=='val':
            return cls.VAL
        raise ValueError(x)

    def __repr__(self):
        if self==self.TRAIN:
            return 'train'
        if self==self.TEST:
            return 'test'
        if self==self.VAL:
            return 'val'

        raise RuntimeError


class SkippableTransformOptions(ABC):
    @abstractmethod
    def serialize(self):
        pass

    @staticmethod
    @property
    @abstractmethod
    def skipped() -> bool:
        pass

@dataclass(frozen=True)
class TransformOptions(SkippableTransformOptions, ABC):
    skipped = False
    def serialize(self):
        return dataclasses.asdict(self)

class SkippedTransform(SkippableTransformOptions):
    skipped = True
    def serialize(self):
        return 'Skipped'

class Transform(ABC):

    @abstractmethod
    def __call__(self, m: np.ndarray, featureTypes: Iterable[Type[Transformable]]) -> np.ndarray:
        pass

class BaseTransform(Transform, ABC):
    @classmethod
    @abstractmethod
    def is_rotation(cls) -> bool:
        pass

    @classmethod
    @abstractmethod
    def global_option_type(cls) -> Type[TransformOptions]:
        pass

    @classmethod
    @abstractmethod
    def default_global_options(cls, phase: Phase) -> SkippableTransformOptions:
        pass

    @abstractmethod
    def _call(self, m: np.ndarray, global_opt: TransformOptions, featureTypes: List[Type[Transformable]]) -> np.ndarray:
        pass

    @classmethod
    def read_global_options(cls, global_options_conf: Mapping[str, Any], phase:Phase) -> SkippableTransformOptions:
        keys = [x.lower() for x in global_options_conf.keys()]
        assert all([x in ['train', 'test', 'val'] for x in keys])

        key = repr(phase)
        if key in global_options_conf:
            value = global_options_conf[key]
            if isinstance(value, str) and value.lower() == 'skipped':
                return SkippedTransform()
            if isinstance(value, dict):
                return cls.global_option_type()(**value)
            raise ValueError(value)
        return cls.default_global_options(phase)

    @classmethod
    def validate_global_options(cls, global_options_conf: Mapping[str, Any]) -> Mapping[str, Any]:

        assert all([x in ['train', 'test', 'val'] for x in global_options_conf.keys()])
        allOptions = {}

        for phase in Phase:
            opt: SkippableTransformOptions = cls.read_global_options(global_options_conf, phase)
            allOptions[repr(phase)] = opt.serialize()

        return allOptions

    @classmethod
    def validate_options(cls, options_conf: Mapping[str, Mapping[str, Any]]):
        return cls.validate_global_options(options_conf)

    def set_seed(self, seed:int) -> None:
        self.generator.manual_seed(seed)

    def __init__(self, options_conf: Mapping[str, Any], phase: Phase, generator:MyGenerator, **kwargs):
        self.global_options = self.read_global_options(options_conf,phase)
        self.generator = generator

    @profile
    def __call__(self, m: np.ndarray, featureTypes: List[Type[Transformable]]) -> np.ndarray:
        assert m.ndim == 4
        assert m.shape[0] == len(featureTypes)

        if self.global_options.skipped:
            return m
        else:
            assert isinstance(self.global_options, TransformOptions)
            ret: np.ndarray = self._call(m, self.global_options, featureTypes)

        assert m.shape == ret.shape
        return ret



def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


class LocalTransform(BaseTransform, ABC):

    @classmethod
    @abstractmethod
    def local_option_type(cls) -> Type[TransformOptions]:
        pass

    @classmethod
    @abstractmethod
    def default_local_options(cls, phase: Phase, ft: Type[Transformable]) -> SkippableTransformOptions:
        pass

    @abstractmethod
    def makeCallableSequence(self, global_opt: TransformOptions) -> Iterable[
        Callable[[np.ndarray, TransformOptions, int], np.ndarray]]:
        pass

    def __init__(self, options_conf: Mapping[str, Mapping[str, Any]], phase: Phase, generator:MyGenerator):

        assert all(x in ['local','global'] for x in options_conf.keys())
        local_options_conf = options_conf.get('local', {})
        global_options_conf = options_conf.get('global', {})

        local_options = self.read_local_options(local_options_conf, phase)
        def get_local_options(t: Type[Transformable]):
            return local_options.get(t, self.default_local_options(phase, t))
        self.get_local_options = get_local_options

        super().__init__(global_options_conf, phase, generator)

    @classmethod
    def read_local_options(cls, local_options_conf: Mapping[str, Mapping[str,Any]], phase: Phase) \
            -> Mapping[Type[Transformable],SkippableTransformOptions]:

        keys = [x.lower() for x in local_options_conf.keys()]
        assert all([x in ['train','test','val'] for x in keys])

        ret = {}
        key = repr(phase).lower()
        if key in local_options_conf:
            assert isinstance(local_options_conf[key], dict)
            for featureName, opt in local_options_conf[key].items():
                featureType = get_feature_cls(featureName)
                if isinstance(opt, str) and opt.lower() == 'skipped':
                    ret[featureType] = SkippedTransform()
                elif isinstance(opt, dict):
                    t = cls.local_option_type()
                    ret[featureType] = t(**opt)
                else:
                    raise ValueError(opt)
        return ret

    @classmethod
    def validate_local_options(cls, local_options_conf: Mapping[str, Mapping[str,Any]]) -> Mapping[str, Mapping[str,Any]]:
        assert all(key.lower() in ['train','val','test'] for key in local_options_conf.keys())

        allOptions = {}
        for phase in Phase:
            allOptions[repr(phase)] = {}

            local_options: Mapping[Type[Transformable], SkippableTransformOptions] = \
                cls.read_local_options(local_options_conf, phase)

            for featureType in all_subclasses(Transformable):
                if not inspect.isabstract(featureType):
                    featureName = featureType.__name__
                    if featureName == 'ComposedFeatures':
                        continue
                    assert featureName not in allOptions[repr(phase)]
                    allOptions[repr(phase)][featureName] = local_options.get(featureType, cls.default_local_options(phase, featureType)).serialize()

        return allOptions

    @classmethod
    def validate_options(cls,options_conf: Mapping[str, Mapping[str, Any]]):
        assert all(x in ['local', 'global'] for x in options_conf.keys())
        local_options_conf = options_conf.get('local', {})
        global_options_conf = options_conf.get('global', {})
        return {
            'local': cls.validate_local_options(local_options_conf),
            'global': cls.validate_global_options(global_options_conf)
        }

    @profile
    def _call(self, m: np.ndarray, global_opt: TransformOptions, featureTypes: List[Type[Transformable]]) -> np.ndarray:

        for fun in self.makeCallableSequence(global_opt):
            channels = []
            for c, featureType in zip(range(m.shape[0]), featureTypes):
                opt = self.get_local_options(featureType)
                if opt.skipped:
                    channels.append(m[c])
                else:
                    m3d = fun(m[c], opt, c)
                    assert m3d.shape == m[c].shape
                    channels.append(m3d)
            m = np.stack(channels, axis=0)

        return m


class ComposedTransform(Transform, ABC):
    def __init__(self, transformer_classes:Iterable[Type[BaseTransform]], conf_options:Iterable[Mapping[str,Any]],
                 common_config:Mapping[str,Any], phase:Phase, seed:int, dtype=np.float32, convert_to_torch=True):

        self.dtype = dtype
        self.convert_to_torch = convert_to_torch
        self.transforms = []

        args = []

        for i,(cls, options_conf) in enumerate(zip(transformer_classes,conf_options)):
            iseed = seed+i
            config = {**common_config,
                      **{'generator': MyGenerator().manual_seed(iseed)}
                      }
            args.append((cls, options_conf, phase, iseed))
            self.transforms.append(cls(options_conf=options_conf, phase=phase, **config))

        self.state = common_config, self.dtype, self.convert_to_torch, args

    # FIXME This is probably bugged if we pickle after calling the first time
    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        logger.warning(f'Pickling the {type(self).__name__} instance - This has not been properly tested')
        self.transforms = []
        common_config, self.dtype, self.convert_to_torch, args = state
        for i, (cls, options_conf, phase, iseed) in enumerate(args):
            config = {**common_config,
                      **{'generator': MyGenerator().manual_seed(iseed)}
                      }
            self.transforms.append(cls(options_conf=options_conf, phase=phase, **config))

    # FIXME Should also update the state
    def set_seeds(self, seed:int) -> None:
        for i,t in enumerate(self.transforms):
            t.set_seed(seed + i)

    @profile
    def __call__(self, m: np.ndarray, featureTypes: List[Type[Transformable]]) -> Union[torch.Tensor, np.ndarray]:
        assert m.ndim == 4
        for trans in self.transforms:
            m = trans(m, featureTypes)
        assert m.ndim == 4
        if self.convert_to_torch:
            return torch.from_numpy(m.astype(dtype=self.dtype))
        else:
            return m
