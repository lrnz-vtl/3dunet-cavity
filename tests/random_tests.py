import unittest

from pytorch3dunet.datasets.featurizer import get_features, ComposedFeatures
from pytorch3dunet.augment.transforms import Phase
from pytorch3dunet.augment.utils import Transformer


class TestRandom(unittest.TestCase):
    '''
    Test that same random transoformations are applied to different features
    '''

    def __init__(self):
        features_config = [{'name': 'PotentialGrid'}, {'name': 'KalasantyFeatures'}, {'name': 'AtomLabel'}]
        channels: ComposedFeatures = get_features(features_config)

        = [{'name': 'PotentialGrid'}, {'name': 'KalasantyFeatures'}, {'name': 'AtomLabel'},
           {'name': 'LabelClass'}]

        channels: ComposedFeatures = get_features(config)

    def test_seeds(self):
        transformer_config = [
            {'name': 'RandomRotate3D',
             'local': {
                 'train': {
                     'AtomLabel':'skipped'
                 }
             }},
            {'name': 'RandomRotate3D'}
        ]
        phase = Phase.TRAIN
        self.transformer = Transformer(transformer_config=transformer_config, common_config={},
                                   allowRotations=True, debug_str='', stats=None)
        self.transformer.validate()
        self.raw_transform = self.transformer.create_transform(phase, '_raw')
        self.label_transform = self.transformer.create_transform(phase, '_label')


if __name__ == '__main__':
    unittest.main()
