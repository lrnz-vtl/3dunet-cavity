import torch
import yaml
from pathlib import Path
from pytorch3dunet.datasets.utils import get_class
from pytorch3dunet.unet3d.utils import get_logger

num_workers = 0

testpath = Path(rf"C:\Users\loren\Data") / "Test"
trainpath = Path(rf"C:\Users\loren\Data") / "Train"
valpath = Path(rf"C:\Users\loren\Data") / "Val"
checkpointpath = Path(rf"C:\Users\loren\Data") / "Checkpoint"

logger = get_logger('TrainingSetup')

def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))

def load_config():
    config = _load_config_yaml(rf"C:\Users\loren\pytorch-3dunet\resources\3DUnet_lightsheet_boundary\train_config.yml")

    config['loaders']['train']['file_paths'] = [str(trainpath)]
    config['loaders']['val']['file_paths'] = [str(valpath)]
    config['trainer']['checkpoint_dir'] = str(checkpointpath)

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

    config = load_config()
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

