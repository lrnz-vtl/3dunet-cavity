from __future__ import annotations
import torch
from torch.utils.data import DataLoader, ConcatDataset
from pytorch3dunet.unet3d.utils import get_logger, profile
from pytorch3dunet.datasets.base import AbstractDataset, default_prediction_collate
from pytorch3dunet.datasets.utils import _loader_classes
from pytorch3dunet.datasets.config import LoadersConfig

MAX_SEED = 2 ** 32 - 1

logger = get_logger('Loaders')


def get_train_loaders(config, loaders_config: LoadersConfig):
    features_config = config['featurizer']
    transformer_config = config['transformer']

    logger.info('Creating training and validation set loaders...')

    dataset_class: AbstractDataset = _loader_classes(loaders_config.dataset_cls_str)

    num_workers = loaders_config.num_workers
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.batch_size

    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for train/val loader: {batch_size}')

    if hasattr(dataset_class, 'collate_fn'):
        collate_fn = dataset_class.collate_fn
    else:
        # Will use standard collate function
        collate_fn = None

    train_datasets = dataset_class.create_datasets(loaders_config=loaders_config, features_config=features_config,
                                                   transformer_config=transformer_config, phase='train')

    def train_dataloader_gen():
        for dataset in train_datasets:
            seed = torch.randint(size=(1,), high=MAX_SEED).item()
            dataset.set_transform_seeds(seed)
        return DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          collate_fn=collate_fn, pin_memory=True)

    val_datasets = dataset_class.create_datasets(loaders_config=loaders_config, features_config=features_config,
                                                 transformer_config=transformer_config, phase='val')
    # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
    val_Dataloader = DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False,
                                num_workers=num_workers,
                                collate_fn=collate_fn, pin_memory=True)

    def val_dataloader_gen():
        return val_Dataloader

    return {
        'train': train_dataloader_gen,
        'val': val_dataloader_gen
    }


def get_test_loaders(config):
    """
    Returns test DataLoader.

    :return: generator of DataLoader objects
    """

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    features_config = config['featurizer']
    transformer_config = config['transformer']

    logger.info('Creating test set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warn(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class: AbstractDataset = _loader_classes(dataset_cls_str)

    test_datasets = dataset_class.create_datasets(loaders_config, features_config,
                                                  transformer_config=transformer_config, phase='test')

    num_workers = loaders_config.get('num_workers', 0)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for dataloader: {batch_size}')

    # use generator in order to create data loaders lazily one by one
    for test_dataset in test_datasets:
        logger.info(f'Loading test set from: {test_dataset.name}...')
        if hasattr(test_dataset, 'prediction_collate'):
            collate_fn = test_dataset.prediction_collate
        else:
            collate_fn = default_prediction_collate

        yield DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                         collate_fn=collate_fn, pin_memory=True)
