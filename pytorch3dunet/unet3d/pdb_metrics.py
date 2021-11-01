import numpy as np
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.datasets.utils_pdb import PdbDataHandler
from pytorch3dunet.unet3d.metrics import MeanIoU
from sklearn.metrics import f1_score

logger = get_logger('EvalMetric')


class PocketScore:
    """
    Generic class to represent scores evaluated on the residue predictions rather than grid
    """
    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def _callSingle(self, input, pdbObj : PdbDataHandler):
        pred = pdbObj.makePdbPrediction(input)
        structure, _ = pdbObj.getStructureLigand()
        pocket = pdbObj.genPocket()

        if len(pred) == 0:
            # TODO Is this necessary?
            prednums = set()
        else:
            prednums = set(t.getResnum() for t in pred.iterResidues())

        truenums = set(t.getResnum() for t in pocket.iterResidues())
        allnums = set(t.getResnum() for t in structure.iterResidues())

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        y_true = []
        y_pred = []

        for num in allnums:
            if num not in prednums and num not in truenums:
                tn += 1
                y_pred.append(-1)
                y_true.append(-1)
            elif num not in prednums and num in truenums:
                fn += 1
                y_pred.append(-1)
                y_true.append(1)
            elif num in prednums and num not in truenums:
                fp += 1
                y_pred.append(1)
                y_true.append(-1)
            elif num in prednums and num in truenums:
                tp += 1
                y_pred.append(1)
                y_true.append(1)
            else:
                raise Exception

        return self.scoref(y_true, y_pred)

    def __call__(self, input, target, pdbData):
        if isinstance(pdbData, list) and len(pdbData)>1:
            return np.mean([self._callSingle(input[i], pdbData[i]) for i in range(len(pdbData))])
        return self._callSingle(input[0], pdbData[0])


class PocketFScore(PocketScore):
    @staticmethod
    def scoref(*args):
        return f1_score(*args)


class MixedGridPdbScore:
    def __init__(self, pdbWeight:float = 0.5):
        assert pdbWeight > 0 and pdbWeight < 1
        self.pdbWeight = pdbWeight
        self.pdbScore = PocketFScore()
        self.gridScore = MeanIoU()

    def __call__(self, input, target, pdbObj=None):
        return self.pdbWeight*self.pdbScore(input,target,pdbObj) +\
               (1-self.pdbWeight)*self.gridScore(input,target,pdbObj)