from pytorch3dunet.augment.transforms import Phase
from pytorch3dunet.datasets.featurizer import get_feature_cls
from typing import Mapping, Any


class ExpectedChange:
    def __init__(self, conf):
        self.expectedChange = self.convertExpected(conf)

    @staticmethod
    def convertExpected(d: Mapping[str, Any]):
        ret = {}
        for phase, value in d.items():
            phase = Phase.from_str(phase)
            if isinstance(value, bool):
                ret[phase] = value
            else:
                assert isinstance(value, dict)
                ret[phase] = {}
                for clsname, expected_change in value.items():
                    ret[phase][get_feature_cls(clsname)] = expected_change
        return ret

    def is_change_expected(self, phase:Phase, channel_type:type):
        if isinstance(self.expectedChange[phase], bool):
            return self.expectedChange[phase]
        elif isinstance(self.expectedChange[phase], dict):
            return self.expectedChange[phase][channel_type]
        else:
            raise ValueError