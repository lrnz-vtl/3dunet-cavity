"""
Hacky way to generate the tmp data folder, to perform analysis later
"""

import torch
import yaml
from pathlib import Path
from pytorch3dunet.datasets.utils_pdb import PdbDataHandler
from pytorch3dunet.unet3d.utils import get_logger
from argparse import ArgumentParser
import os

logger = get_logger('TrainingSetup')

def load_config(runconfigPath, nworkers, device):

    runconfig = yaml.safe_load(open(runconfigPath, 'r'))

    dataFolder = Path(runconfig['dataFolder'])
    runFolder = Path(runconfig.get('runFolder', Path(runconfigPath).parent))

    train_config = runFolder / 'train_config.yml'

    config = yaml.safe_load(open(train_config, 'r'))

    config['loaders']['train']['file_paths'] = [str(dataFolder / name) for name in runconfig['train']]
    config['loaders']['num_workers'] = nworkers
    config['loaders']['tmp_folder'] = str(runFolder / 'tmp')
    config['loaders']['pdb2pqrPath'] = runconfig.get('pdb2pqrPath', 'pdb2pqr')

    config['dry_run'] = runconfig.get('dryRun', False)

    os.makedirs(config['loaders']['tmp_folder'], exist_ok=True)

    # Get a device to train on
    if device is not None:
        config['device'] = device

    device_str = 'cpu'
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

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)

    logger.info(f'Batch size for train/val loader: {batch_size}')

    datasets = PdbDataHandler.create_datasets(loaders_config, phase='train')
    # dataLoader = DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=True, num_workers=num_workers)
