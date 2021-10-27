from contextlib import nullcontext as nc
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
from pytorch3dunet.datasets.loaders import get_train_loaders
from pytorch3dunet.datasets.config import RunConfig
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric, get_log_metrics
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import profile, get_logger, \
    get_tensorboard_formatter, create_sample_plotter, create_optimizer, \
    create_lr_scheduler, get_number_of_learnable_parameters
from pytorch3dunet.datasets.featurizer import BaseFeatureList, get_features
from pytorch3dunet.augment.utils import Transformer
from typing import Mapping
import torch
from torch.profiler import record_function
# from torch import autocast
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from . import utils

logger = get_logger('UNet3DTrainer')


def create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, dry_run,
                   dump_inputs, log_criterions, run_config: RunConfig):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    # get tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # get sample plotter
    sample_plotter = create_sample_plotter(trainer_config.pop('sample_plotter', None))

    # start training from scratch
    return UNet3DTrainer(model=model,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         loss_criterion=loss_criterion,
                         eval_criterion=eval_criterion,
                         log_criterions=log_criterions,
                         device=config['device'],
                         loaders=loaders,
                         tensorboard_formatter=tensorboard_formatter,
                         sample_plotter=sample_plotter,
                         dry_run=dry_run,
                         dump_inputs=dump_inputs,
                         run_config=run_config,
                         **trainer_config)


def build_trainer(config: Mapping, run_config: RunConfig):
    features: BaseFeatureList = get_features(config['featurizer'])

    transformer = Transformer(transformer_config=config['transformer'], common_config={}, allowRotations=True)
    transformer.validate()

    model = get_model(features=features, model_config=config['model'])
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create log metrics
    log_criterions = get_log_metrics(config)

    # Create data loaders
    loaders = get_train_loaders(config=config, runconfig=run_config)

    # Create the optimizer
    optimizer = create_optimizer(config['optimizer'], model)

    # Create learning rate adjustment strategy
    lr_scheduler = create_lr_scheduler(config.get('lr_scheduler', None), optimizer)

    # Create model trainer
    trainer = create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                             loss_criterion=loss_criterion, eval_criterion=eval_criterion,
                             log_criterions=log_criterions,
                             loaders=loaders, dry_run=config['dry_run'], dump_inputs=config['dump_inputs'],
                             run_config=run_config)

    return trainer


class UNet3DTrainer:

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,
                 run_config: RunConfig,
                 log_criterions=None,
                 max_num_epochs=100, max_num_iterations=int(1e5),
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 tensorboard_formatter=None, sample_plotter=None,
                 skip_train_validation=False,
                 dry_run=False,
                 dump_inputs=False):

        self.run_config = run_config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        # Additional List of scores to log just for logging, not for validating
        if log_criterions is None:
            self.log_criterions = []
        else:
            self.log_criterions = log_criterions

        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.dry_run = dry_run
        self.dump_inputs = dump_inputs

        self.valLoaders = self.loaders['val']()

        logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter
        self.sample_plotter = sample_plotter

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        self.skip_train_validation = skip_train_validation

        if run_config.mixed:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    @profile
    def fit(self):

        for i in range(self.num_epoch, self.max_num_epochs):

            logger.info(f'Entering training epoch {i}')
            trainLoaders = self.loaders['train']()

            with record_function("3dunet-train") if self.run_config.profile else nc():
                should_terminate = self.train(trainLoaders)

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epoch += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    @profile
    def train(self, trainLoaders):
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()

        for t in trainLoaders:
            self.optimizer.zero_grad()

            logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                        f'Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            with record_function("3dunet-split_training_batch") if self.run_config.profile else nc():
                names, pdbObjs, (input, target, weight) = self._split_training_batch(t)
            logger.debug(f'Forward samples {names}')

            if self.dump_inputs:
                self.save_inputs(names, input, target)

            if self.dry_run:
                continue

            with autocast() if self.run_config.mixed else nc() as ac:
            # with autocast(self.device.type) if self.run_config.mixed else nc() as ac:
                if ac is not None:
                    logger.debug(f"Autocast from {ac.prev} to {ac.fast_dtype} for {ac.device}")
                with record_function("3dunet-forward_pass") if self.run_config.profile else nc():
                    output = self.model(input)
                    loss = self.loss_criterion(output, target)

            train_losses.update(loss.item(), self._batch_size(input))

            # compute gradients and update parameters
            with record_function("3dunet-optimize") if self.run_config.profile else nc():

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            if self.num_iterations % self.validate_after_iters == 0:
                with record_function("3dunet-validate") if self.run_config.profile else nc():
                    self.validate_step()

            if self.num_iterations % self.log_after_iters == 0:
                with record_function("3dunet-validate") if self.run_config.profile else nc():
                    self.log_step(input, output, target, train_eval_scores, train_losses)

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()
        log_scores = {type(log_criterion).__name__: utils.RunningAverage() for log_criterion in self.log_criterions}

        if self.sample_plotter is not None:
            self.sample_plotter.update_current_dir()

        with torch.no_grad():

            for i, t in enumerate(self.valLoaders):
                logger.info(f'Validation iteration {i}')

                names, pdbObjs, (input, target, weight) = self._split_training_batch(t)

                if self.dry_run:
                    continue

                with autocast(self.device.type) if self.run_config.mixed else nc() as ac:
                    if ac is not None:
                        logger.debug(f"Autocast to {ac.fast_dtype}")
                    output = self.model(input)
                    loss = self.loss_criterion(output, target)

                val_losses.update(loss.item(), self._batch_size(input))

                # if model contains final_activation layer for normalizing logits apply it, otherwise
                # the evaluation metric will be incorrectly computed
                if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                    output = self.model.final_activation(output)

                if i % 100 == 0:
                    self._log_images(input, target, output, 'val_')

                eval_score = self.eval_criterion(output, target, pdbObjs)
                val_scores.update(eval_score.item(), self._batch_size(input))

                for log_criterion in self.log_criterions:
                    name = type(log_criterion).__name__
                    log_score = log_criterion(output, target, pdbObjs)
                    log_scores[name].update(log_score.item(), self._batch_size(input))

                if self.sample_plotter is not None:
                    self.sample_plotter(i, input, output, target, 'val')

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            self._log_stats('val', val_losses.avg, val_scores.avg,
                            {name: log_score.avg for name, log_score in log_scores.items()})
            logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            return val_scores.avg

    def save_inputs(self, names, input, target):
        dump_dir = f'{self.checkpoint_dir}/dumps'
        os.makedirs(dump_dir, exist_ok=True)
        for name, inp, targ in zip(names, input, target):
            targ = targ[0]
            h5path = f'{dump_dir}/{name}.h5'
            with h5py.File(h5path, 'w') as h5:
                h5.create_dataset('labels', data=targ)
                for i, arr in enumerate(inp):
                    h5.create_dataset(f'raws_{i}', data=arr)

    def validate_step(self):
        # set the model in eval mode
        self.model.eval()
        # evaluate on validation set
        eval_score = self.validate()
        # set the model back to training mode
        self.model.train()

        # adjust learning rate if necessary
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(eval_score)
        else:
            self.scheduler.step()
        # log current learning rate in tensorboard
        self._log_lr()
        # remember best validation metric
        is_best = self._is_best_eval_score(eval_score)

        # save checkpoint
        self._save_checkpoint(is_best)

    def log_step(self, input, output, target, train_eval_scores, train_losses):
        # if model contains final_activation layer for normalizing logits apply it, otherwise both
        # the evaluation metric as well as images in tensorboard will be incorrectly computed
        if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
            output = self.model.final_activation(output)

        # compute eval criterion
        if not self.skip_train_validation:
            eval_score = self.eval_criterion(output, target)
            train_eval_scores.update(eval_score.item(), self._batch_size(input))

        # log stats, params and images
        logger.info(
            f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
        self._log_stats('train', train_losses.avg, train_eval_scores.avg, {})
        self._log_params()
        self._log_images(input, target, output, 'train_')

    @profile
    def _split_training_batch(self, t):
        names, pdbObjs, t = t

        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return names, pdbObjs, (input, target, weight)

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters,
            'skip_train_validation': self.skip_train_validation
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg, log_scores_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }
        tag_value.update({f'{phase}_{name}_avg': log_score_avg for name, log_score_avg in log_scores_avg.items()})

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, prefix=''):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations, dataformats='CHW')

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
