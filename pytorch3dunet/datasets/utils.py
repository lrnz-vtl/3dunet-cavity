import importlib
from pytorch3dunet.unet3d.utils import get_logger, profile
import torch
import collections

MAX_SEED = 2 ** 32 - 1

logger = get_logger('Utils')


def default_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def get_class(class_name, modules):
    for module in modules:
        try:
            m = importlib.import_module(module)
            clazz = getattr(m, class_name, None)
            if clazz is not None:
                return clazz
        except ImportError:
            logger.warning(f'Could not import module {module}')
    raise RuntimeError(f'Unsupported dataset class: {class_name}, or its module could not be imported')


def _loader_classes(class_name):
    modules = [
        'pytorch3dunet.datasets.pdb',
        'pytorch3dunet.datasets.random',
        'pytorch3dunet.datasets.utils'
    ]
    return get_class(class_name, modules)


def get_slice_builder(raws, labels, config):
    if config is None:
        config = {}
        slice_builder_cls = TrivialSliceBuilder
    else:
        assert 'name' in config
        slice_builder_cls = _loader_classes(config['name'])
    return slice_builder_cls(raws, labels, **config)


class TrivialSliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the the patch and stride shape
    """

    def __init__(self, raw_dataset, label_dataset, **kwargs):
        """
        :param raw_datasets: ndarray of raw data
        :param label_datasets: ndarray of ground truth labels
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param kwargs: additional metadata
        """

        self._raw_slices = self._build_slices(raw_dataset)
        if label_dataset is None:
            self._label_slices = None
        else:
            # take the first element in the label_datasets to build slices
            self._label_slices = self._build_slices(label_dataset)
            assert len(self._raw_slices) == len(self._label_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @staticmethod
    def _build_slices(dataset):
        assert dataset.ndim == 4
        in_channels, i_z, i_y, i_x = dataset.shape
        return [(slice(0, in_channels), slice(0, i_z), slice(0, i_y), slice(0, i_x))]
