import os
import torch.nn as nn
from pytorch3dunet.datasets.loaders import get_test_loaders
from pytorch3dunet.datasets.config import RunConfig
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import profile, get_logger, get_number_of_learnable_parameters
from pytorch3dunet.datasets.featurizer import BaseFeatureList, get_features
from pytorch3dunet.augment.utils import Transformer
from typing import Mapping
import torch
from . import utils
import importlib
from pathlib import Path

logger = get_logger('UNet3DTester')


def _get_predictor(model, config):
    config = dict(config)
    predictor_config = config.pop('predictor')
    output_dir = predictor_config.pop('output_dir')
    class_name = 'PdbPredictor'

    ms = [
        importlib.import_module('pytorch3dunet.unet3d.pdb_predictor')
    ]
    for m in ms:
        if hasattr(m, class_name):
            predictor_class = getattr(m, class_name)
            return predictor_class(model, output_dir, config, **predictor_config)
    raise AttributeError(f"Predictor {class_name} not found in modules")


def run_predictions(config: Mapping, run_config: RunConfig):
    features: BaseFeatureList = get_features(config['featurizer'])

    transformer = Transformer(transformer_config=config['transformer'], common_config={}, allowRotations=True)
    transformer.validate()

    model = get_model(features=features, model_config=config['model'])
    # use DataParallel if more than 1 GPU available
    device = config['device']
    gpus_to_use = 0
    if torch.cuda.device_count() >= 1 and not device.type == 'cpu':
        if run_config.max_gpus is None:
            gpus_to_use = torch.cuda.device_count()
        else:
            gpus_to_use = min(torch.cuda.device_count(), run_config.max_gpus)
        if gpus_to_use > 1:
            if gpus_to_use != torch.cuda.device_count():
                raise NotImplemented
            model = nn.DataParallel(model)
        logger.info(f'Using {gpus_to_use} GPUs for training')

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    checkpoint_dir = config['trainer']['checkpoint_dir']
    model_path = str(Path(checkpoint_dir) / "best_checkpoint.pytorch")

    utils.load_checkpoint(model_path, model)

    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(device)

    output_dir = config['predictor'].get('output_dir', None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving predictions to: {output_dir}')

    # create predictor instance
    predictor = _get_predictor(model, config)

    for test_loader in get_test_loaders(config=config, runconfig=run_config, gpus_to_use=gpus_to_use):
        # run the model prediction on the test_loader and save the results in the output_dir
        predictor(test_loader)
