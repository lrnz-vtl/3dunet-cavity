import torch
import yaml
from pathlib import Path
from pytorch3dunet.datasets.utils import get_class
from pytorch3dunet.unet3d.utils import get_logger
from argparse import ArgumentParser
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# testpath = Path(rf"C:\Users\loren\deep_apbs") / "test_sub"
# trainpath = Path(rf"C:\Users\loren\deep_apbs") / "train_sub"
# valpath = Path(rf"C:\Users\loren\deep_apbs") / "val_sub"
checkpointname = "checkpoint"
base_config = "train_config_base.yml"

logger = get_logger('TrainingSetup')

def load_config(runconfig, nworkers):
    config = yaml.safe_load(open(base_config, 'r'))

    dataFolder = Path(runconfig['dataFolder'])
    runFolder = Path(runconfig['runFolder'])

    config['loaders']['train']['file_paths'] = [str(dataFolder / name) for name in runconfig['train']]
    config['loaders']['val']['file_paths'] = [str(dataFolder / name) for name in runconfig['val']]

    config['loaders']['num_workers'] = nworkers

    config['trainer']['checkpoint_dir'] = str(runFolder / checkpointname)

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

    args = parser.parse_args()
    runconfig = args.runconfig
    nworkers = int(args.numworkers)

    config = load_config(yaml.safe_load(open(runconfig,'r')), nworkers)
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

