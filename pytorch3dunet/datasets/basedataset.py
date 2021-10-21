from __future__ import annotations
from pathlib import Path
from multiprocessing import Lock
import h5py
import numpy as np
from pytorch3dunet.augment.utils import Transformer
from pytorch3dunet.augment.transforms import Transform, Phase
from pytorch3dunet.augment.standardize import Stats
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.datasets.featurizer import Transformable, LabelClass
from pytorch3dunet.datasets.utils import default_prediction_collate, get_slice_builder
from typing import Iterable, Type
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

logger = get_logger('AbstractDataset')
lock = Lock()


class AbstractDataset(Dataset, ABC):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    @classmethod
    @abstractmethod
    def create_datasets(cls, dataset_config, features_config, transformer_config, phase) -> Iterable[AbstractDataset]:
        pass

    def set_transform_seeds(self, seed: int) -> None:
        self.raw_transform.set_seeds(seed)
        self.label_transform.set_seeds(seed)

    def __init__(self,
                 name: str,
                 raws: np.ndarray, labels: np.ndarray,
                 featuresTypes: Iterable[Type[Transformable]],
                 tmp_data_folder,
                 phase: Phase,
                 slice_builder_config,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
                 random_seed=0,
                 allowRotations=True,
                 debug_str=None):

        self.name = name

        self.tmp_data_folder = tmp_data_folder
        self.h5path = Path(self.tmp_data_folder) / f'grids.h5'
        self.featureTypes: Iterable[Type[Transformable]] = featuresTypes

        labels = np.expand_dims(labels, axis=0)

        if phase in [Phase.TRAIN, phase.VAL]:
            mirror_padding = None

        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"

        self.mirror_padding = mirror_padding
        self.phase = phase

        self.stats = Stats(raws)

        self.transformer = Transformer(transformer_config=transformer_config, common_config={},
                                       allowRotations=allowRotations, debug_str=debug_str, stats=self.stats)
        self.raw_transform = self.transformer.create_transform(self.phase, '_raw')

        if phase != Phase.TEST:
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.create_transform(self.phase, '_label')

            self._check_dimensionality(raws, labels)
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            labels = None

            # add mirror padding if needed
            if self.mirror_padding is not None:
                raise NotImplementedError('mirror_padding branch of the code has not been tested')
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                channels = [np.pad(r, pad_width=pad_width, mode='reflect') for r in raw]
                padded_volume = np.stack(channels)

                raws = padded_volume

        # build slice indices for raw and label data sets
        slice_builder = get_slice_builder(raws, labels, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices

        with h5py.File(self.h5path, 'w') as h5:
            h5.create_dataset('raws', data=raws)
            if labels is not None:
                h5.create_dataset('labels', data=labels)

        self.patch_count = len(self.raw_slices)
        logger.debug(f'Number of patches: {self.patch_count}')

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        with h5py.File(self.h5path, 'r') as h5:
            raws = np.array(h5['raws'])

            # get the slice for a given index 'idx'
            raw_idx = self.raw_slices[idx]
            # get the raw data patch for a given slice
            raw_patch_transformed = self.raw_transform(raws[raw_idx], self.featureTypes)

            if self.phase == Phase.TEST:
                # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
                if len(raw_idx) == 4:
                    raw_idx = raw_idx[1:]
                return (raw_patch_transformed, raw_idx)
            else:
                labels = np.array(h5['labels']).astype("<f8")
                # get the slice for a given index 'idx'
                label_idx = self.label_slices[idx]
                label_patch_transformed = self.label_transform(labels[label_idx], [LabelClass])

                # return the transformed raw and label patches
                return raw_patch_transformed, label_patch_transformed

    def _transform_patches(self, dataset, label_idx, transformer: Transform):
        transformed_patch = transformer(dataset[label_idx], self.featureTypes)
        return transformed_patch

    def __len__(self):
        return self.patch_count

    @staticmethod
    def _check_dimensionality(raws, labels):
        assert raws.ndim == 4
        assert labels.ndim == 4

        assert raws.shape[1:] == labels.shape[1:]

    @classmethod
    def prediction_collate(cls, batch):
        """Default collate_fn. Override in child class for non-standard datasets."""
        return default_prediction_collate(batch)
