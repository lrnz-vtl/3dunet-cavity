import importlib

import numpy as np
import torch
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from pytorch3dunet.datasets.utils import calculate_stats
from torchvision.transforms import Compose
from pytorch3dunet.unet3d.utils import get_logger
from typing import Dict, Union, Optional
from numbers import Number
from scipy.spatial.transform import Rotation
from abc import ABC, abstractmethod


MAX_SEED = 2**32 - 1
# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
GLOBAL_RANDOM_STATE = np.random.RandomState(47)

logger = get_logger('Transforms')

numpy_to_torch_dtype_dict = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}
torch_to_numpy_dtype_dict = {y:x for x,y in numpy_to_torch_dtype_dict.items()}


class PicklableGenerator(torch.Generator):
    def __getstate__(self):
        return self.get_state()
    def __setstate__(self, state):
        return self.set_state(state)

    def genSeed(self):
        return torch.randint(generator=self, high=MAX_SEED, size=(1,)).item()


class SampleStats:
    def __init__(self, raws):
        ndim = raws[0].ndim
        self.min, self.max, self.mean, self.std = calculate_stats(raws)
        logger.debug(f"mean={self.mean}, std={self.std}")
        self.channelStats = None
        if ndim == 4:
            channels = raws[0].shape[0]
            self.channelStats = [SampleStats([raw[i] for raw in raws]) for i in range(channels)]



class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, m):
        raise NotImplementedError


class RandomFlip(BaseTransform):
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, generator: PicklableGenerator, axis_prob=0.5, **kwargs):
        self.random_state = np.random.RandomState(seed=generator.genSeed())
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        for axis in self.axes:
            if self.random_state.uniform() > self.axis_prob:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m





# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=0 when transforming the labels
class ElasticDeformation(BaseTransform):
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, generator, spline_order, alpha=2000, sigma=50, execution_probability=0.1, apply_3d=True,
                 **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        :param apply_3d: if True apply deformations in each axis
        """
        self.random_state = np.random.RandomState(seed=generator.genSeed())
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability
        self.apply_3d = apply_3d

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim in [3, 4]

            if m.ndim == 3:
                volume_shape = m.shape
            else:
                volume_shape = m[0].shape

            if self.apply_3d:
                dz = gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="reflect") * self.alpha
            else:
                dz = np.zeros_like(m)

            dy, dx = [
                gaussian_filter(
                    self.random_state.randn(*volume_shape),
                    self.sigma, mode="reflect"
                ) * self.alpha for _ in range(2)
            ]

            z_dim, y_dim, x_dim = volume_shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx

            if m.ndim == 3:
                return map_coordinates(m, indices, order=self.spline_order, mode='reflect')
            else:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode='reflect') for c in m]
                return np.stack(channels, axis=0)

        return m


class Standardize(BaseTransform):
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    """

    def __init__(self, stats: SampleStats, eps=1e-10, channels:Optional[Dict[str,Number]] = None,  **kwargs):
        """ If channels is None average across all channel. Otherwise channel-wise separately to each channel in channels
        """
        assert stats is not None
        self.stats = stats
        if channels is None:
            self.instructions = None
        else:
            self.instructions = {c['axis']: c for c in channels}
        self.eps = eps

    def __call__(self, m):

        if self.instructions is None:
            mean, std = self.stats.mean, self.stats.std
        else:
            assert m.ndim == 4
            mean = np.zeros(m.shape[0])
            std = np.zeros(m.shape[0])
            for i in range(m.shape[0]):
                if i in self.instructions.keys():
                    if 'stats' in self.instructions[i].keys():
                        mean[i] = float(self.instructions[i]['stats']['mean'])
                        std[i] = float(self.instructions[i]['stats']['std'])
                    else:
                        mean[i] = self.stats.channelStats[i].mean
                        std[i] = self.stats.channelStats[i].std
                else:
                    mean[i] = 0
                    std[i] = 1

            mean = mean[:,np.newaxis,np.newaxis,np.newaxis]
            std = std[:, np.newaxis, np.newaxis, np.newaxis]

        return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)


class PercentileNormalizer(BaseTransform):
    def __init__(self, pmin, pmax, channelwise=False, eps=1e-10, **kwargs):
        self.eps = eps
        self.pmin = pmin
        self.pmax = pmax
        self.channelwise = channelwise

    def __call__(self, m):
        if self.channelwise:
            axes = list(range(m.ndim))
            # average across channels
            axes = tuple(axes[1:])
            pmin = np.percentile(m, self.pmin, axis=axes, keepdims=True)
            pmax = np.percentile(m, self.pmax, axis=axes, keepdims=True)
        else:
            pmin = np.percentile(m, self.pmin)
            pmax = np.percentile(m, self.pmax)

        return (m - pmin) / (pmax - pmin + self.eps)


class Normalize(BaseTransform):
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, min_value, max_value, **kwargs):
        assert max_value > min_value
        self.min_value = min_value
        self.value_range = max_value - min_value

    def __call__(self, m):
        norm_0_1 = (m - self.min_value) / self.value_range
        return np.clip(2 * norm_0_1 - 1, -1, 1)


class AdditiveGaussianNoise(BaseTransform):
    def __init__(self, generator, scale=(0.0, 1.0), execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = np.random.RandomState(seed=generator.genSeed())
        self.scale = scale

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            std = self.random_state.uniform(self.scale[0], self.scale[1])
            logger.debug(f"Adding gaussian noise with std = {std}")
            logger.debug(f"Type before noise: {m.dtype}")
            gaussian_noise = self.random_state.normal(0, std, size=m.shape)
            if torch.is_tensor(m):
                # TODO This is slow, should define it directly on the GPU
                gaussian_noise = torch.from_numpy(gaussian_noise.astype(torch_to_numpy_dtype_dict[m.dtype]))
            else:
                gaussian_noise = gaussian_noise.astype(m.dtype)
            logger.debug(f"Type of noise: {gaussian_noise.dtype}")
            ret = m + gaussian_noise
            logger.debug(f"Type after noise: {ret.dtype}")
            return ret
        return m


class AdditivePoissonNoise(BaseTransform):
    def __init__(self, generator, lam=(0.0, 1.0), execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = np.random.RandomState(seed=generator.genSeed())
        self.lam = lam

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            lam = self.random_state.uniform(self.lam[0], self.lam[1])
            poisson_noise = self.random_state.poisson(lam, size=m.shape)
            return m + poisson_noise
        return m


class ToTensor(BaseTransform):
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        # FIXME Can it be avoided to define on GPU first
        return torch.from_numpy(m.astype(dtype=self.dtype))


class Identity(BaseTransform):
    def __init__(self, **kwargs):
        pass

    def __call__(self, m):
        return m


class LabelToTensor(BaseTransform):
    def __call__(self, m):
        m = np.array(m)
        return torch.from_numpy(m.astype(dtype='int64'))


def get_transformer(config, allowRotations, stats, debug_str=None):
    base_config = {'stats': stats, 'allowRotations': allowRotations, 'debug_str':debug_str}
    return Transformer(config, base_config)


class Transformer:
    def __init__(self, phase_config, base_config):
        self.phase_config = phase_config
        self.config_base = base_config
        self.seed = torch.randint(MAX_SEED, size=(1,)).item()
        # GLOBAL_RANDOM_STATE.randint(MAX_SEED)

        logger.debug(f'Initialising new random state with seed = {self.seed}')

    def raw_transform(self):

        return self._create_transform('raw')

    def label_transform(self):
        return self._create_transform('label')

    def weight_transform(self):
        return self._create_transform('weight')

    @staticmethod
    def _transformer_class(class_name):
        m = importlib.import_module('pytorch3dunet.augment.transforms')
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self, name):
        assert name in self.phase_config, f'Could not find {name} transform'
        return Compose([
            self._create_augmentation(c) for c in self.phase_config[name]
        ])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        # FIXME Object needs to be reinitialised between epochs, all this will be always the same
        config['generator'] = PicklableGenerator().manual_seed(self.seed)
        aug_class = self._transformer_class(config['name'])
        return aug_class(**config)