from __future__ import annotations
import torch
from typing import Type
from torch.utils.data import DataLoader, ConcatDataset
from pytorch3dunet.unet3d.utils import get_logger, profile
from pytorch3dunet.datasets.base import AbstractDataset, default_prediction_collate
from pytorch3dunet.datasets.utils import _loader_classes
from pytorch3dunet.datasets.config import RunConfig

MAX_SEED = 2 ** 32 - 1

logger = get_logger('Loaders')


def get_train_loaders(config, runconfig: RunConfig, gpus_to_use: int):
    loaders_config = runconfig.loaders_config
    features_config = config['featurizer']
    transformer_config = config['transformer']

    logger.info('Creating training and validation set loaders...')

    dataset_class: AbstractDataset = _loader_classes(loaders_config.dataset_cls_str)

    num_workers = loaders_config.num_workers
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.batch_size

    if config['device'].type != 'cpu':
        assert gpus_to_use >= 1
        logger.info(
            f'Will use {gpus_to_use} GPUs. Using batch_size = {gpus_to_use} * {batch_size}')
        batch_size = batch_size * gpus_to_use

    logger.info(f'Batch size for train/val loader: {batch_size}')

    if hasattr(dataset_class, 'collate_fn'):
        collate_fn = dataset_class.collate_fn
    else:
        # Will use standard collate function
        collate_fn = None

    train_datasets = dataset_class.create_datasets(loaders_config=loaders_config, pdb_workers=runconfig.pdb_workers,
                                                   features_config=features_config,
                                                   transformer_config=transformer_config, phase='train')

    def train_dataloader_gen():
        for dataset in train_datasets:
            seed = torch.randint(size=(1,), high=MAX_SEED).item()
            dataset.set_transform_seeds(seed)
        return DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          collate_fn=collate_fn, pin_memory=loaders_config.pin_memory)

    val_datasets = dataset_class.create_datasets(loaders_config=loaders_config, pdb_workers=runconfig.pdb_workers,
                                                 features_config=features_config, transformer_config=transformer_config,
                                                 phase='val')
    # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
    val_Dataloader = DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False,
                                num_workers=num_workers,
                                collate_fn=collate_fn, pin_memory=loaders_config.pin_memory)

    def val_dataloader_gen():
        return val_Dataloader

    return {
        'train': train_dataloader_gen,
        'val': val_dataloader_gen
    }


def get_test_loaders(config, runconfig: RunConfig, gpus_to_use: int):
    loaders_config = runconfig.loaders_config
    features_config = config['featurizer']
    transformer_config = config['transformer']

    logger.info('Creating training and validation set loaders...')

    dataset_class: Type[AbstractDataset] = _loader_classes(loaders_config.dataset_cls_str)

    num_workers = loaders_config.num_workers
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.batch_size

    if config['device'].type != 'cpu':
        assert gpus_to_use >= 1
        logger.info(
            f'Will use {gpus_to_use} GPUs. Using batch_size = {gpus_to_use} * {batch_size}')
        batch_size = batch_size * gpus_to_use

    logger.info(f'Batch size for test loader: {batch_size}')

    if hasattr(dataset_class, 'prediction_collate'):
        collate_fn = dataset_class.prediction_collate
    else:
        collate_fn = default_prediction_collate

    test_datasets = dataset_class.create_datasets(loaders_config=loaders_config, pdb_workers=runconfig.pdb_workers,
                                                  features_config=features_config,
                                                  transformer_config=transformer_config,
                                                  phase='test')

    return DataLoader(ConcatDataset(test_datasets), batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, collate_fn=collate_fn, pin_memory=loaders_config.pin_memory)