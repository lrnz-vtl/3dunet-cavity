from pytorch3dunet.unet3d.utils import get_logger, Phase
from pytorch3dunet.unet3d.config import parse_args
from pytorch3dunet.datasets.utils_pdb import PdbDataHandler
from pytorch3dunet.datasets.featurizer import ComposedFeatures, get_features

logger = get_logger("Cache Generator")

if __name__ == '__main__':

    args, config, run_config = parse_args()
    assert run_config.loaders_config.dataset_cls_str == 'PDBDataset'
    assert run_config.loaders_config.data_config.gridscache is not None
    logger.debug(f'Read Config is: {config}')

    pdb_config = run_config.loaders_config.data_config

    features_config = config['featurizer']

    features: ComposedFeatures = get_features(features_config)

    def makeRawsLabels(x: PdbDataHandler):
        return x.getRawsLabels(features=features, grid_size=run_config.loaders_config.grid_size,
                               ligand_mask_radius=pdb_config.ligand_mask_radius)

    PdbDataHandler.map_datasets(loaders_config=run_config.loaders_config, pdb_workers=run_config.pdb_workers,
                                features_config=features_config, transformer_config=config['transformer'],
                                phases=[phase for phase in Phase], f=makeRawsLabels,
                                generating_cache=True)
