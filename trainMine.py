import torch
import yaml
from pathlib import Path
from pytorch3dunet.datasets.utils import get_class
from pytorch3dunet.unet3d.utils import get_logger, str2bool
from argparse import ArgumentParser

checkpointname = "checkpoint"

base_config_default = "train_config_base.yml"
base_config_test = "train_config_test.yml"

logger = get_logger('TrainingSetup')

def load_config(runconfig, nworkers, device, test):
    runconfig = yaml.safe_load(open(runconfig, 'r'))

    if test:
        base_config = base_config_test
    else:
        base_config = base_config_default

    config = yaml.safe_load(open(base_config, 'r'))

    dataFolder = Path(runconfig['dataFolder'])
    runFolder = Path(runconfig['runFolder'])

    config['loaders']['train']['file_paths'] = [str(dataFolder / name) for name in runconfig['train']]
    config['loaders']['val']['file_paths'] = [str(dataFolder / name) for name in runconfig['val']]

    config['loaders']['num_workers'] = nworkers

    suf = ''
    if test:
        suf = '_test'

    config['trainer']['checkpoint_dir'] = str(runFolder / (checkpointname + suf))

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
    parser.add_argument("-t", "--test", type=str2bool, nargs='?', const=True, default=False,
                        help="Test run.")

    args = parser.parse_args()
    runconfig = args.runconfig
    nworkers = int(args.numworkers)

    config = load_config(runconfig, nworkers, args.device, args.test)
    logger.info(config)

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

    # Start training
    trainer.fit()

