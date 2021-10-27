import torch
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.unet3d.config import parse_args
import contextlib
from pathlib import Path
from pytorch3dunet.unet3d.trainer import build_trainer, UNet3DTrainer
from torch.profiler import profile as torch_profile, tensorboard_trace_handler

if __name__ == '__main__':

    args, config, run_config = parse_args()

    logger = get_logger('TrainingSetup')
    logger.debug(f'Read Config is: {config}')

    manual_seed = config.get('manual_seed', None)

    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = run_config.benchmark
        logger.info(f'Setting torch.backends.cudnn.benchmark={torch.backends.cudnn.benchmark}')

    trainer: UNet3DTrainer = build_trainer(config, run_config)

    logdir = str(Path(config['trainer']['checkpoint_dir']) / 'profile')
    if run_config.profile:
        logger.info(f'Saving profile logs to {logdir}')

    if args.schedule:
        schedule = torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1)
    else:
        schedule = None
    if args.memory:
        profile_memory = True
    else:
        profile_memory = False
    if args.nostack:
        with_stack = False
    else:
        with_stack = True

    with torch_profile(on_trace_ready=tensorboard_trace_handler(logdir), profile_memory=profile_memory,
                       schedule=schedule, with_stack=with_stack) if args.profile \
            else contextlib.nullcontext() as prof:
        trainer.fit()
