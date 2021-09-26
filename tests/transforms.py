import unittest
from pytorch3dunet.augment.standardize import Standardize
from pytorch3dunet.augment.randomRotate import RandomRotate3D
from pytorch3dunet.augment.globalTransforms import RandomFlip


class TestStringMethods(unittest.TestCase):

    def test_validate_options(self):
        Standardize.validate_options(options_conf={})
        RandomRotate3D.validate_options(options_conf={})
        RandomFlip.validate_options(options_conf={})


if __name__ == '__main__':
    unittest.main()
