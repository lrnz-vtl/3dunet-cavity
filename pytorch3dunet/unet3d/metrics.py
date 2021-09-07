import importlib
import numpy as np
import torch
from pytorch3dunet.unet3d.losses import compute_per_channel_dice
from pytorch3dunet.unet3d.utils import get_logger, expand_as_one_hot
from pytorch3dunet.datasets.utils_pdb import PdbDataHandler

logger = get_logger('EvalMetric')


class PdbFScore:
    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def _callSingle(self, input, pdbObj : PdbDataHandler):
        s = pdbObj.makePdbPrediction(input)
        

    def __call__(self, input, target, pdbData):
        if isinstance(pdbData, list):
            return np.mean([self._callSingle(input[i], pdbData[i]) for i in len(pdbData)])
        return self._callSingle(input, pdbData)


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, thres=0.5, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels
        self.thres = thres

    def __call__(self, input, target, pdbObj=None):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        assert input.size() == target.size()

        per_batch_iou = []
        for _input, _target in zip(input, target):
            binary_prediction = self._binarize_predictions(_input, n_classes, self.thres)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))

    def _binarize_predictions(self, input, n_classes, thres):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input >= thres
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target, pdbObj=None):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


def get_evaluation_metric(config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    def _metric_class(class_name):
        m = importlib.import_module('pytorch3dunet.unet3d.metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)
