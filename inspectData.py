from potsim2 import PotGrid
import os
import prody as pr
import numpy as np
from multiprocessing import Pool
import miniball

folder = 'runs/gen_data/tmp'

# name = os.listdir(folder)[0]

def run(name):
    struct_fname = f'{folder}/{name}/structure.pdb'
    structure: pr.AtomGroup = pr.parsePDB(struct_fname)

    coords = np.array(list(structure.getCoords()))

    mb = miniball.Miniball(coords)
    C = mb.center()
    R = np.sqrt(mb.squared_radius())

    grid_fname = f'{folder}/{name}/grid.dx.gz'

    if R > 80:
        print(f'{name} cannot fit in a ball', C, R)
    elif os.path.exists(grid_fname):
        grid = PotGrid(struct_fname, grid_fname)

        CubeC = np.array([(e[-1]+e[0])/2 for e in grid.edges])

        dists = np.sqrt(((coords - CubeC)**2).sum(axis=1))
        if max(dists) > 80:
            print(f'{name} need to center Cube')
    else:
        print(f'{name} grid does not exist')

pool = Pool(processes=12)
pool.map(run, os.listdir(folder))