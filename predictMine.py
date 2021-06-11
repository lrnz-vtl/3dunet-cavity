import importlib
import os
import torch
import torch.nn as nn
from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model
from argparse import ArgumentParser
import yaml
from pathlib import Path

logger = utils.get_logger('UNet3DPredict')

checkpointname = "checkpoint"

predname = 'predictions'

test_config_default = "test_config_base.yml"
train_config_default = "train_config_base.yml"
test_config_test = "test_config_test.yml"
train_config_test = "train_config_test.yml"


def load_config(runconfig, nworkers, device, test):
    runconfig = yaml.safe_load(open(runconfig, 'r'))

    if test:
        test_config = test_config_test
        train_config = train_config_test
    else:
        test_config = test_config_default
        train_config = train_config_default

    config = yaml.safe_load(open(test_config, 'r'))
    train_config = yaml.safe_load(open(train_config, 'r'))

    dataFolder = Path(runconfig['dataFolder'])
    runFolder = Path(runconfig['runFolder'])

    suf = ''
    if test:
        suf = '_test'

    config['loaders']['output_dir'] = str(runFolder / (predname + suf))

    config['loaders']['test']['file_paths'] = [str(dataFolder / name) for name in runconfig['test']]

    config['loaders']['num_workers'] = nworkers

    checkpoint_dir = runFolder / checkpointname
    config['model_path'] = str(checkpoint_dir / "best_checkpoint.pytorch")

    # Copy model from train conf
    config['model'] = train_config['model']

    # Get a device to train on
    if device is not None:
        config['device'] = device
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
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('pytorch3dunet.unet3d.predictor')
    predictor_class = getattr(m, class_name)

    return predictor_class(model, output_dir, config, **predictor_config)



def main():
    parser = ArgumentParser()
    parser.add_argument("-r", "--runconfig", dest='runconfig', type=str, required=True,
                        help=f"The run config yaml file")
    parser.add_argument("-n", "--numworkers", dest='numworkers', type=int, required=True,
                        help=f"Number of workers")
    parser.add_argument("-d", "--device", dest='device', type=str, required=False,
                        help=f"Device")
    parser.add_argument("-t", "--test", type=utils.str2bool, nargs='?', const=True, default=False,
                        help="Test run.")

    args = parser.parse_args()
    runconfig = args.runconfig
    nworkers = int(args.numworkers)

    # Load configuration
    config = load_config(runconfig, nworkers, args.device, args.test)

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