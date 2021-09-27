import unittest

import numpy as np

from pytorch3dunet.augment.transforms import PicklableGenerator, Phase
from pytorch3dunet.augment.standardize import Standardize, Stats
from pytorch3dunet.augment.randomRotate import RandomRotate3D
from pytorch3dunet.augment.globalTransforms import RandomFlip
from pytorch3dunet.datasets.featurizer import get_features, ComposedFeatures, Transformable, get_feature_cls
from typing import List, Type, Mapping, Any


class TestTransform(unittest.TestCase):
    seed = 0
    N = 100

    def __init__(self, *args, **kwargs):
        config = [{'name': 'PotentialGrid'}, {'name': 'KalasantyFeatures'}, {'name': 'AtomLabel'}, {'name':'LabelClass'}]

        channels: ComposedFeatures = get_features(config)
        self.channel_types: List[Type[Transformable]] = channels.feature_types
        assert channels.num_features == len(self.channel_types)
        self.input = np.random.random(size=(channels.num_features, self.N,self.N,self.N))
        super().__init__(*args, **kwargs)

    def _convertExpected(self, d:Mapping[str, Any]):
        ret = {}
        for phase, value in d.items():
            phase = Phase.from_str(phase)
            if isinstance(value,bool):
                ret[phase] = value
            else:
                assert isinstance(value,dict)
                ret[phase] = {}
                for clsname, expected_change in value.items():
                    ret[phase][get_feature_cls(clsname)] = expected_change
        return ret

    @staticmethod
    def _is_change_expected(expectedChange:Any, channel_type):
        if isinstance(expectedChange, bool):
            return expectedChange
        elif isinstance(expectedChange, dict):
            return expectedChange[channel_type]
        else:
            raise ValueError

    def _test_transform(self, cls, options_conf, expectedChange, common_config):
        expectedChange = self._convertExpected(expectedChange)
        cls.validate_options(options_conf)

        outputs = {}

        for phase in Phase:
            common_config = dict(common_config)
            common_config['generator'] = PicklableGenerator().manual_seed(self.seed)
            transform = cls(options_conf=options_conf, phase=phase, **common_config)
            outputs[phase] = transform(self.input, self.channel_types)
            assert self.input.shape == outputs[phase].shape

            for i,channel_type in enumerate(self.channel_types):
                o3d = outputs[phase][i]
                i3d = self.input[i]
                isExpectedChange = self._is_change_expected(expectedChange[phase], channel_type)
                assert isExpectedChange != np.all(o3d == i3d), \
                    f"transform={cls.__name__}, isExpectedChange={isExpectedChange}, phase={phase}, i={i}, channel_type={channel_type}"
        return outputs

    def test_flip(self):
        expectedChange = {
            'train': True,
            'test': False,
            'val': False,
        }
        options_conf = {'train': {'axis_prob': 1.0}}
        common_config = {}
        cls = RandomFlip
        self._test_transform(cls, options_conf, expectedChange,common_config)

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
        outputs = self._test_transform(cls, options_conf, expectedChange, common_config)

        expectedChange = self._convertExpected(expectedChange)
        for phase in Phase:
            for i, channel_type in enumerate(self.channel_types):
                isExpectedChange = self._is_change_expected(expectedChange[phase], channel_type)
                mean = outputs[phase][i].mean()
                std = outputs[phase][i].std()
                if isExpectedChange:
                    assert np.isclose(mean, 0)
                    assert np.isclose(std, 1)

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
