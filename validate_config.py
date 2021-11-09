import logging
from pytorch3dunet.unet3d.config import parse_args
from pytorch3dunet.unet3d.utils import get_logger, set_default_log_level

logger = get_logger('TrainingSetup')

if __name__ == '__main__':

    args, config, class_config = parse_args()

    if args.debug:
        set_default_log_level(logging.DEBUG)

    logger.debug(f'Read Config is: {config}')