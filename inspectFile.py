from potsim2 import PotGrid
from pytorch3dunet.datasets.utils_pdb import PdbDataHandler
import os
import prody as pr
import miniball
import numpy as np

loaders_config = {
    'dataset': 'StandardPDBDataset',
    'grid_config': {'grid_size': 160, 'ligand_mask_radius': 4.5},
    'featurizer': [{'name': 'PotentialGrid'}],
    'train': {
        'transformer': {'label': [], 'raw': []},
        'file_paths': ['/home/lorenzo/deep_apbs/srcData/pdbbind_v2019_refined/1ii5']
    },
    'tmp_folder': 'runs/gen_data/tmp',
    'pdb2pqrPath': '/home/lorenzo/pymol/bin/pdb2pqr',
    'num_workers': 1
}

def f(raws, labels):
    return None

results = PdbDataHandler.map_datasets(loaders_config, phase='train', f=f)

folder = 'runs/gen_data/tmp'
name = '5jzi'

struct_fname = f'{folder}/{name}/structure.pdb'

print('Reading protein...')
structure: pr.AtomGroup = pr.parsePDB(struct_fname)
print('Read protein.')

# grid_fname = f'{folder}/{name}/grid.dx.gz'
# grid = PotGrid(struct_fname, grid_fname)

print('Forming numpy array...')
S = np.array(list(structure.getCoords()))
print('Formed numpy array.')

print('Calculating miniball...')

mb = miniball.Miniball(S)
C = mb.center()
R = np.sqrt(mb.squared_radius())
print('Center', C)
print('Radius', R)