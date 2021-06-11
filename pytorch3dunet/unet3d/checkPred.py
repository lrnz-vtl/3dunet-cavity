from IPython.core.display import display, HTML
import h5py
import torch
import numpy as np
from pytorch3dunet.unet3d.losses import *
from pytorch3dunet.unet3d.metrics import MeanIoU
from pathlib import Path
import glob

basepred = Path('/home/lorenzo/3dunet-cavity/runs/run_210601_local/predictions')
baseorig = Path('/home/lorenzo/deep_apbs/destData/pdbbind_v2013_core_set_0')  # /2yfe/2yfe_grids.h5'

class DiceProbLoss(DiceLoss):
    def __init__(self):
        super(DiceProbLoss, self).__init__(normalization='none')

initLosses = {
    "BCE": nn.BCELoss,
    "Dice": DiceProbLoss,
    "MeanIoU": MeanIoU
}

def genDataSets():
    for predfname in glob.glob(str(basepred / '*_grids_predictions.h5')):
        name = Path(predfname).name.split('_')[0]
        labelfname = baseorig / name / f'{name}_grids.h5'

        labelT = torch.tensor(h5py.File(labelfname)['label'], dtype=torch.float32)
        labelT = labelT[None, None]
        predT = torch.tensor(h5py.File(predfname)['predictions'])
        predT = predT[None]

        yield (predT, labelT)

class BCEDiceProbLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEDiceProbLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.beta = beta
        self.dice = DiceProbLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)

class RunningAverage:
    def __init__(self, loss):
        self.count = 0
        self.sum = 0
        self.loss = loss

    def update(self, pred, label):
        self.count += 1
        self.sum += self.loss(pred, label).item()

    def value(self):
        return self.sum / self.count


class AverageLosses:
    def __init__(self, losses):
        self.losses = {name: RunningAverage(loss()) for name, loss in losses.items()}

    def update(self, pred, label):
        for name in self.losses.keys():
            self.losses[name].update(pred, label)

    def value(self):
        return {name: loss.value() for name, loss in self.losses.items()}





def main():
    OracleLoss = AverageLosses(initLosses)
    UnetLoss = AverageLosses(initLosses)
    RandLoss = AverageLosses(initLosses)
    RandUnitLoss = AverageLosses(initLosses)
    ZeroLoss = AverageLosses(initLosses)
    UnitLoss = AverageLosses(initLosses)

    for predT, labelT in genDataSets():
        constPred = torch.zeros_like(labelT)
        ZeroLoss.update(constPred, labelT)

        constPred[:] = 1
        UnitLoss.update(constPred, labelT)

        UnetLoss.update(predT, labelT)

        OracleLoss.update(labelT, labelT)

        randPred = torch.rand(size=constPred.size(), dtype=constPred.dtype, device=constPred.device, requires_grad=False)
        RandLoss.update(randPred, labelT)

        randPred[randPred < 0.5] = 0
        randPred[randPred > 0.5] = 1
        RandUnitLoss.update(randPred, labelT)

if __name__ == '__main__':
    main()