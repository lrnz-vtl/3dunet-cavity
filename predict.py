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

def load_config(runconfigPath, nworkers, device):
    runconfig = yaml.safe_load(open(runconfigPath, 'r'))

    dataFolder = Path(runconfig['dataFolder'])
    runFolder = Path(runconfig.get('runFolder', Path(runconfigPath).parent))

    train_config = Path(runFolder) / 'train_config.yml'
    train_config = yaml.safe_load(open(train_config, 'r'))

    test_config = Path(runFolder) / 'test_config.yml'
    config = yaml.safe_load(open(test_config, 'r'))

    config['loaders']['output_dir'] = str(runFolder / predname)

    config['loaders']['test']['file_paths'] = [str(dataFolder / name) for name in runconfig['test']]
    config['loaders']['tmp_folder'] = str(runFolder / 'tmp_predict')
    config['loaders']['pdb2pqrPath'] = runconfig.get('pdb2pqrPath', 'pdb2pqr')

    config['loaders']['num_workers'] = nworkers

    for key,val in train_config['loaders'].items():
        if key not in ['val','train'] and key not in config['loaders']:
            config['loaders'][key] = val

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

    ms = [importlib.import_module('pytorch3dunet.unet3d.predictor'),
          importlib.import_module('pytorch3dunet.unet3d.pdb_predictor')
          ]
    predictor_class = None
    for m in ms:
        try:
            predictor_class = getattr(m, class_name)
        except AttributeError:
            pass
    if predictor_class is None:
        raise AttributeError("predictor not found in modules")

    return predictor_class(model, output_dir, config, **predictor_config)



def main():
    parser = ArgumentParser()
    parser.add_argument("-r", "--runconfig", dest='runconfig', type=str, required=True,
                        help=f"The run config yaml file")
    parser.add_argument("-n", "--numworkers", dest='numworkers', type=int, required=True,
                        help=f"Number of workers")
    parser.add_argument("-d", "--device", dest='device', type=str, required=False,
                        help=f"Device")

    args = parser.parse_args()
    runconfig = args.runconfig
    nworkers = int(args.numworkers)

    config = load_config(runconfig, nworkers, args.device)
    logger.debug(f'Read Config is: {config}')

    # Create the model
    config['model']['in_channels'] = len(config['loaders']['featurizer'])
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