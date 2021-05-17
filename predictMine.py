import importlib
import os

import torch
import torch.nn as nn

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model

import yaml
from pathlib import Path

num_workers = 8

logger = utils.get_logger('UNet3DPredict')

base = Path(rf"C:\Users\loren\cavityPred")
testpath = base / "train_sub"
predpath = base / "train_sub_pred"
checkpointpath = base / "checkpoint_sub2"
base_config = rf"C:\Users\loren\pytorch-3dunet\resources\3DUnet_lightsheet_boundary\test_config_Lorenzo.yml"


def load_config():
    config = yaml.safe_load(open(base_config, 'r'))

    config['loaders']['output_dir'] = str(predpath)
    config['loaders']['test']['file_paths'] = [str(testpath)]
    config['loaders']['num_workers'] = num_workers
    config['model_path'] = checkpointpath / "best_checkpoint.pytorch"

    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warn('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    return config

def _get_predictor(model, output_dir, config):
    predictor_config = load_config()
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('pytorch3dunet.unet3d.predictor')
    predictor_class = getattr(m, class_name)

    del predictor_config['model']

    return predictor_class(model, output_dir, config, **predictor_config)


def main():
    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config['model'])

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

    logger.info(f"Sending the model to '{device}'")
    model = model.to(device)

    output_dir = config['loaders'].get('output_dir', None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving predictions to: {output_dir}')

    # create predictor instance
    predictor = _get_predictor(model, output_dir, config)

    for test_loader in get_test_loaders(config):
        # run the model prediction on the test_loader and save the results in the output_dir
        predictor(test_loader)


if __name__ == '__main__':
    main()