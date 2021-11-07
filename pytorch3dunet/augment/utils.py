from pytorch3dunet.augment.transforms import ComposedTransform, Phase, logger, MAX_SEED, BaseTransform
from pytorch3dunet.augment.standardize import Stats
import importlib
import torch
from typing import Iterable, Any, Mapping, Type, Optional
import pprint


def _transformer_class(class_name) -> Type[BaseTransform]:
    modules = [
        'pytorch3dunet.augment.standardize',
        'pytorch3dunet.augment.randomRotate',
        'pytorch3dunet.augment.globalTransforms',
        'pytorch3dunet.augment.trivialRandom',
        'pytorch3dunet.augment.denoise'
    ]
    modules = [importlib.import_module(m) for m in modules]
    for m in modules:
        if hasattr(m, class_name):
            return getattr(m, class_name)
    raise AttributeError(f"Class {class_name} not found in modules")


class Transformer:

    def __init__(self, transformer_config: Iterable[Mapping[str, Any]], common_config: Mapping[str, Any],
                 allowRotations: bool, stats: Optional[Stats] = None, debug_str=None):

        self.common_config = {**common_config, **{'featureStats': stats, 'debug_str': debug_str}}
        self.seed = torch.randint(MAX_SEED, size=(1,)).item()

        self.transformer_classes = []
        self.conf_options = []

        for conf in transformer_config:
            name = conf['name']
            conf = {k: v for k, v in conf.items() if k != 'name'}

            transformer_class = _transformer_class(name)

            if not allowRotations and transformer_class.is_rotation():
                logger.info(f'Removing {transformer_class.__name__} because rotation turned off')
            else:
                self.transformer_classes.append(transformer_class)
                self.conf_options.append(conf)

    def validate(self):
        for cls, options in zip(self.transformer_classes, self.conf_options):
            pp = pprint.pformat(cls.validate_options(options), indent=4)
            logger.info(f'{cls.__name__} options: \n{pp}')

    def create_transform(self, phase: Phase, debug_str='', convert_to_torch=True):
        ''' Needs to be called separately for raw and label '''
        common_config = dict(self.common_config)
        common_config['debug_str'] = common_config['debug_str'] + debug_str
        return ComposedTransform(transformer_classes=self.transformer_classes, conf_options=self.conf_options,
                                 common_config=common_config, phase=phase, seed=self.seed, convert_to_torch=convert_to_torch)


def take_while_deterministic(transformer_config: Iterable[Mapping[str, Any]]):
    det_config = []
    rand_config = list(transformer_config)
    for conf in transformer_config:
        name = conf['name']
        transformer_class = _transformer_class(name)
        if transformer_class.is_random():
            break
        else:
            det_config.append(rand_config.pop(0))
    return det_config, rand_config

