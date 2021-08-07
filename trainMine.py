import sys

import torch
import yaml
from pathlib import Path
from pytorch3dunet.datasets.utils import get_class
from pytorch3dunet.unet3d.utils import get_logger
from argparse import ArgumentParser
import os

checkpointname = "checkpoint"

logger = get_logger('TrainingSetup')

def load_config(runconfigPath, nworkers, device):

    runconfig = yaml.safe_load(open(runconfigPath, 'r'))

    dataFolder = Path(runconfig['dataFolder'])
    runFolder = Path(runconfig.get('runFolder', Path(runconfigPath).parent))

    train_config = runFolder / 'train_config.yml'

    config = yaml.safe_load(open(train_config, 'r'))

    config['loaders']['train']['file_paths'] = [str(dataFolder / name) for name in runconfig['train']]
    config['loaders']['val']['file_paths'] = [str(dataFolder / name) for name in runconfig['val']]

    config['loaders']['num_workers'] = nworkers
    config['loaders']['tmp_folder'] = str(runFolder / 'tmp')
    config['loaders']['pdb2pqrPath'] = runconfig.get('pdb2pqrPath', 'pdb2pqr')
    config['loaders']['tmpl_dir'] = runconfig.get('tmpl_dir')

    os.makedirs(config['loaders']['tmp_folder'], exist_ok=True)

    config['trainer']['checkpoint_dir'] = str(runFolder / checkpointname)

    # Get a device to train on
    if device is not None:
        config['device'] = device

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

if __name__=='__main__':

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

    manual_seed = config.get('manual_seed', None)

    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # create trainer
    trainer_builder_class = 'UNet3DTrainerBuilder'
    trainer_builder = get_class(trainer_builder_class, modules=['pytorch3dunet.unet3d.trainer'])
    trainer = trainer_builder.build(config)

    # sys.exit(0)
    # Start training
    trainer.fit()

