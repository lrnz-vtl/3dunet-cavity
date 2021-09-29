from pytorch3dunet.augment.transforms import ComposedTransform, Phase, logger, MAX_SEED, BaseTransform
from pytorch3dunet.augment.standardize import Stats
import importlib
import torch
from typing import Iterable, Any, Mapping, Type, Optional
import pprint

class Transformer:
    modules = [
        'pytorch3dunet.augment.standardize',
        'pytorch3dunet.augment.randomRotate',
        'pytorch3dunet.augment.globalTransforms'
    ]
    modules = [importlib.import_module(m) for m in modules]

    def __init__(self, transformer_config: Iterable[Mapping[str, Any]], common_config: Mapping[str, Any],
                 allowRotations: bool, stats: Optional[Stats] = None, debug_str=None):

        self.common_config = {**common_config, **{'featureStats': stats, 'debug_str': debug_str}}
        self.seed = torch.randint(MAX_SEED, size=(1,)).item()

        self.transformer_classes = []
        self.conf_options = []

        for conf in transformer_config:
            name = conf['name']
            conf = {k: v for k, v in conf.items() if k != 'name'}

            transformer_class = self._transformer_class(name)

            if not allowRotations and transformer_class.is_rotation():
                logger.info(f'Removing {transformer_class.__name__} because rotation turned off')
            else:
                self.transformer_classes.append(transformer_class)
                self.conf_options.append(conf)

    def validate(self):
        for cls, options in zip(self.transformer_classes, self.conf_options):
            pp = pprint.pformat(cls.validate_options(options), indent=4)
            logger.info(f'{cls.__name__} options: {pp}')

    def _transformer_class(self, class_name) -> Type[BaseTransform]:
        clazz: Optional[Type[BaseTransform]] = None
        for m in self.modules:
            try:
                clazz = getattr(m, class_name)
            except AttributeError:
                pass
        if clazz is None:
            raise AttributeError(f"Class {class_name} not found in modules")
        return clazz

    def create_transform(self, phase: Phase, debug_str=''):
        ''' Needs to be called separately for raw and label '''
        common_config = dict(self.common_config)
        common_config['debug_str'] = common_config['debug_str'] + debug_str
        return ComposedTransform(transformer_classes=self.transformer_classes, conf_options=self.conf_options,
                                 common_config=common_config, phase=phase, seed=self.seed)
