import unittest

import numpy as np
from pytorch3dunet.augment.transforms import Phase, BaseTransform, ComposedTransform
from pytorch3dunet.augment.standardize import Standardize, Stats
from pytorch3dunet.augment.randomRotate import RandomRotate3D
from pytorch3dunet.augment.globalTransforms import RandomFlip
from pytorch3dunet.datasets.featurizer import get_features, ComposedFeatures, Transformable
from typing import Type, Mapping, Any, Callable, Optional
from utils import ExpectedChange


class TestTransform(unittest.TestCase):
    seed = 0
    N = 100

    def __init__(self, *args, **kwargs):
        config = [{'name': 'PotentialGrid'}, {'name': 'KalasantyFeatures'}, {'name': 'AtomLabel'}, {'name':'LabelClass'}]

        channels: ComposedFeatures = get_features(config)
        self.channel_types = channels.feature_types
        assert all(issubclass(t, Transformable) for t in self.channel_types)
        assert channels.num_features == len(self.channel_types)
        self.input = np.random.random(size=(channels.num_features, self.N,self.N,self.N))
        super().__init__(*args, **kwargs)


    def _test_transform(self, cls:Type[BaseTransform], options_conf:Mapping, expectedChange:Mapping,
                        common_config:Mapping, validate_fun:Optional[Callable[[np.ndarray,np.ndarray,int],bool]] = None) -> None:
        expectedChange = ExpectedChange(expectedChange)
        cls.validate_options(options_conf)

        common_config = {**common_config, **{'debug_str':None}}

        for phase in Phase:

            transform = ComposedTransform(transformer_classes=[cls], conf_options=[options_conf],
                                      common_config=common_config, phase=phase, seed=self.seed, convert_to_torch=False)
            transform.__setstate__(transform.__getstate__())

            output = transform(self.input, self.channel_types)
            assert self.input.shape == output.shape

            for i,channel_type in enumerate(self.channel_types):
                o3d = output[i]
                i3d = self.input[i]
                isExpectedChange = expectedChange.is_change_expected(phase, channel_type)
                assert isExpectedChange != np.all(o3d == i3d), \
                    f"transform={cls.__name__}, isExpectedChange={isExpectedChange}, phase={phase}, i={i}, channel_type={channel_type}"
                if isExpectedChange and validate_fun is not None:
                    assert validate_fun(i3d, o3d, i)

    def test_flip(self):
        expectedChange = {
            'train': True,
            'test': False,
            'val': False,
        }
        options_conf = {'train': {'axis_prob': 1.0}}
        common_config = {}
        cls = RandomFlip
        def validate_fun(i3d, o3d, idx) -> bool:
            return np.all(np.flip(i3d,axis=0) == o3d)
        self._test_transform(cls=cls, options_conf=options_conf, expectedChange=expectedChange,
                             common_config=common_config, validate_fun=validate_fun)

    def test_standardize(self):

        x = {
                'PotentialGrid': True,
                 'KalasantyFeatures': False,
                 'LabelClass': False,
                 'AtomLabel': False
        }
        expectedChange = {
            'train': x,
            'test': x,
            'val': x,
        }
        options_conf = {}
        common_config = {'featureStats': Stats(self.input)}
        cls = Standardize

        def validate_fun(i3d, o3d, idx) -> bool:
            mean = o3d.mean()
            std = o3d.std()
            return np.all(np.isclose(mean, 0)) and np.all(np.isclose(std, 1))
        self._test_transform(cls=cls, options_conf=options_conf, expectedChange=expectedChange,
                             common_config=common_config, validate_fun=validate_fun)


    def test_rotate(self):
        expectedChange = {
            'train': True,
            'test': False,
            'val': False,
        }
        options_conf = {}
        common_config = {}
        cls = RandomRotate3D
        self._test_transform(cls, options_conf, expectedChange,common_config)

    def test_rotate_2(self):
        expectedChange = {
            'train':
                {'KalasantyFeatures': False,
                 'LabelClass': True,
                 'PotentialGrid': True,
                 'AtomLabel': True
                 },
            'test': False,
            'val': False,
        }
        options_conf = {'local':
                            {'train':
                                 {'KalasantyFeatures': 'skipped'}
                             }
                        }
        common_config = {}
        cls = RandomRotate3D
        self._test_transform(cls, options_conf, expectedChange,common_config)

    def test_rotate_3(self):
        expectedChange = {
            'train': False,
            'test': False,
            'val': False,
        }
        options_conf = {'global':
                            {'train': 'skipped'}
                        }
        common_config = {}
        cls = RandomRotate3D
        self._test_transform(cls, options_conf, expectedChange,common_config)


class TestOptions(unittest.TestCase):

    def test_validate_options(self):
        Standardize.validate_options(options_conf={})
        RandomRotate3D.validate_options(options_conf={})
        RandomFlip.validate_options(options_conf={})


if __name__ == '__main__':
    unittest.main()
