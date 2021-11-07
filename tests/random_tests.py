import unittest

from pytorch3dunet.datasets.featurizer import get_features, ComposedFeatures, LabelClass
from pytorch3dunet.datasets.features import PotentialGrid
from pytorch3dunet.augment.transforms import Phase
import numpy as np
from typing import Mapping, Iterable, Callable
from pytorch3dunet.augment.utils import Transformer, take_while_deterministic
import torch


class TestRandom(unittest.TestCase):
    '''
    Test that same random transoformations are applied to different features
    '''

    N = 100

    def __init__(self, *args, **kwargs):
        features_config = [{'name': 'PotentialGrid'}, {'name': 'KalasantyFeatures'}, {'name': 'AtomLabel'}]
        channels: ComposedFeatures = get_features(features_config)
        self.feature_types = channels.feature_types

        random3darray = np.random.normal(size=(self.N,self.N,self.N))

        self.raws = np.stack([random3darray for _ in range(channels.num_features)])
        self.labels = np.expand_dims(random3darray, axis=0)

        assert self.raws.shape == (channels.num_features, self.N,self.N,self.N)
        assert self.labels.shape == (1, self.N, self.N, self.N)

        super().__init__(*args, **kwargs)

    def validate(self, transformer_config:Iterable[Mapping],
                              validate_label_outputs: Callable[[torch.Tensor, torch.Tensor, type], bool],
                              validate_input_output:Callable[[torch.Tensor,torch.Tensor],bool]
                              ) -> None:
        phase = Phase.TRAIN
        self.transformer = Transformer(transformer_config=transformer_config, common_config={},
                                       allowRotations=True, debug_str='', stats=None)
        self.transformer.validate()
        self.raw_transform = self.transformer.create_transform(phase, '_raw', convert_to_torch=False)
        self.label_transform = self.transformer.create_transform(phase, '_label', convert_to_torch=False)

        raws_transformed = self.raw_transform(self.raws, self.feature_types)
        label_transformed = self.label_transform(self.labels, [LabelClass])[0]

        assert validate_input_output(self.labels[0], label_transformed)

        for feature_type, raw, raw_transformed in zip(self.feature_types, self.raws, raws_transformed):
            assert raw_transformed.shape == (self.N, self.N, self.N)
            assert validate_label_outputs(label_transformed, raw_transformed, feature_type)
            assert validate_input_output(raw, raw_transformed)

    def test_seeds(self):
        transformer_config = [
            {
                'name': 'TrivialTransform',
                'local':
                    {
                        'train':
                            {
                                'LabelClass': 'skipped'
                            }
                    }
            },
            {
                'name': 'RandomRotate3D',
                'local': {
                    'train':
                        {
                            'PotentialGrid': {'mode': 'constant'}
                        }
                }
            }
        ]
        self.validate(transformer_config, lambda x,y,t: np.array_equal(x,y), lambda x, y: not np.array_equal(x, y))


    def test_seeds2(self):
        transformer_config = [
            {
                'name': 'TrivialTransform',
                'local':
                    {
                        'train':
                            {
                                'PotentialGrid': 'skipped'
                            }
                    }
            },
            {
                'name': 'RandomRotate3D',
                'local': {
                    'train':
                        {
                            'PotentialGrid': {'mode': 'constant'}
                        }
                }
            }
        ]
        self.validate(transformer_config, lambda x,y,t: np.array_equal(x,y), lambda x, y: not np.array_equal(x, y))

    def test_control3(self):
        transformer_config = [
            {
                'name': 'TrivialTransform',
                'local':
                    {
                        'train':
                            {
                                'PotentialGrid': 'skipped'
                            }
                    }
            },
            {
                'name': 'RandomRotate3D',
                'global': {'train': 'skipped'},
                'local': {
                    'train':
                        {
                            'KalasantyFeatures': {'mode': 'nearest'}
                        }
                }
            }
        ]
        self.validate(transformer_config, lambda x,y,t: np.array_equal(x,y), lambda x, y: np.array_equal(x, y))

    def test_baseline(self):
        transformer_config = [
            {
                'name': 'RandomRotate3D',
                'local': {
                    'train':
                        {
                            'PotentialGrid': {'mode': 'constant'}
                        }
                }
            }
        ]
        self.validate(transformer_config, lambda x, y, t: np.array_equal(x, y), lambda x, y: not np.array_equal(x, y))

    def test_control(self):
        transformer_config = [
            {
                'name': 'RandomRotate3D'
            }
        ]

        def validate_label_outputs(label,raw,t):
            if t == PotentialGrid:
                return not np.array_equal(label,raw)
            return np.array_equal(label,raw)

        self.validate(transformer_config, validate_label_outputs, lambda x, y: not np.array_equal(x, y))

    def test_control2(self):
        transformer_config = [
            {
                'name': 'RandomFlip',
                'train': {
                    'axis_prob': 1.0
                }
            },
            {
                'name': 'RandomRotate3D'
            }
        ]
        def validate_label_outputs(label,raw,t):
            if t == PotentialGrid:
                return not np.array_equal(label,raw)
            return np.array_equal(label,raw)

        self.validate(transformer_config, validate_label_outputs, lambda x, y: not np.array_equal(x, y))

if __name__ == '__main__':
    unittest.main()
