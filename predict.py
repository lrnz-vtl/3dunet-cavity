from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.unet3d.config import parse_args
from pytorch3dunet.unet3d import tester

if __name__ == '__main__':

    args, config, run_config = parse_args()

    logger = get_logger('TestSetup')
    logger.debug(f'Read Config is: {config}')

    tester.run_predictions(config=config, run_config=run_config)
