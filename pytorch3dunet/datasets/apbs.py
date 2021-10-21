import numpy as np
from typing import List, Type, Iterable, Mapping
from pytorch3dunet.unet3d.utils import get_logger
from potsim2 import PotGrid
import prody as pr
import os
import subprocess
from pathlib import Path

logger = get_logger('Featurizer')

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
GLOBAL_RANDOM_STATE = np.random.RandomState(47)


def apbsInput(pqr_fname, grid_fname, dielec_const, grid_size):
    return f"""read
    mol pqr {pqr_fname}
end
elec name prot
    mg-manual
    mol 1
    dime {grid_size} {grid_size} {grid_size}
    grid 1.0 1.0 1.0
    gcent mol 1
    lpbe
    bcfl mdh
    ion charge 1 conc 0.100 radius 2.0
    ion charge -1 conc 0.100 radius 2.0
    pdie {dielec_const}
    sdie 78.54
    sdens 10.0
    chgm spl2
    srfm smol
    srad 0.0
    swin 0.3
    temp 298.15
    calcenergy total
    calcforce no
    write pot gz {grid_fname}
end"""


class TmpFile:
    def __init__(self, name, cleanup:bool):
        self.name = str(name)
        self.cleanup = cleanup

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup:
            os.remove(self.name)


class ApbsGridCollection:
    ligand_mask_radius_defaults = 6.5
    grid_size_default = 161

    def __init__(self, structure, ligand, grid_config, dielec_const_list,
                 tmp_data_folder: str, reuse_grids: bool, name: str, pdb2pqrPath: str, cleanup: bool):
        """
        pot_grids: Mapping dielectric constant -> PotGrid
        """
        self.tmp_data_folder = tmp_data_folder
        self.reuse_grids = reuse_grids
        self.name = name
        self.pdb2pqrPath = pdb2pqrPath
        self.cleanup = cleanup

        radius = grid_config.get('ligand_mask_radius', self.ligand_mask_radius_defaults)
        self.grid_size = grid_config.get('grid_size', self.grid_size_default)

        pot_grids, labels = self._genGrids(structure, ligand, self.grid_size, radius, dielec_const_list)

        firstGrid: PotGrid = next(iter(pot_grids.values()))
        assert all(pot_grid.grid.shape == firstGrid.grid.shape for pot_grid in pot_grids.values())
        assert all((np.array(pot_grid.edges) == np.array(firstGrid.edges)).all() for pot_grid in pot_grids.values())
        assert all((pot_grid.delta == firstGrid.delta).all() for pot_grid in pot_grids.values())

        assert firstGrid.grid.shape[0] >= self.grid_size

        self.orig_shape = firstGrid.grid.shape
        if self.grid_size < firstGrid.grid.shape[0]:
            logger.warn(
                f"Requested grid_size = {self.grid_size} is smaller than apbs output grid size = {firstGrid.grid.shape[0]}. "
                f"Applying naive cropping!!!")

        self.grids = {k: pot_grid.grid[:self.grid_size, :self.grid_size, :self.grid_size] for k, pot_grid in
                      pot_grids.items()}
        self.edges = [x[:self.grid_size] for x in firstGrid.edges]
        self.delta = firstGrid.delta
        self.shape = next(iter(self.grids.values())).shape

        assert labels.shape == self.orig_shape
        self.labels = labels[:self.grid_size, :self.grid_size, :self.grid_size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Free memory """
        del self.grids
        del self.labels

    def tmp_file(self, name:str):
        return TmpFile(name, self.cleanup)

    def _runApbs(self, dst_pdb_file: str, grid_size: int, dielec_const_list: List[float]) -> Mapping[float, PotGrid]:

        with self.tmp_file(f"{self.tmp_data_folder}/protein.pqr") as pqr_output:

            if self.reuse_grids and os.path.exists(pqr_output):
                pass
            else:
                logger.debug(f'Running pdb2pqr on {self.name}')

                proc = subprocess.Popen(
                    [
                        self.pdb2pqrPath,
                        "--with-ph=7.4",
                        "--ff=PARSE",
                        "--chain",
                        dst_pdb_file,
                        pqr_output
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                cmd_out = proc.communicate()
                if proc.returncode != 0:
                    raise Exception(cmd_out[1].decode())

            grids = {dielec_const: self._runApbsSingle(pqr_output, dst_pdb_file, grid_size, dielec_const) for dielec_const
                     in dielec_const_list}

        return grids

    def _runApbsSingle(self, pqr_output, dst_pdb_file, grid_size, dielec_const) -> PotGrid:

        with self.tmp_file(f"{self.tmp_data_folder}/grid_{dielec_const}.dx.gz") as grid_fname:

            if self.reuse_grids and os.path.exists(grid_fname):
                pass

            else:
                logger.debug(f'Running apbs on {self.name} with dielec_const = {dielec_const}')

                owd = os.getcwd()
                os.chdir(self.tmp_data_folder)

                with self.tmp_file("apbs-in") as apbs_in_fname:
                    grid_fname_in = '.'.join(str(Path(grid_fname).name).split('.')[:-2])
                    input = apbsInput(pqr_fname=str(Path(pqr_output).name), grid_fname=grid_fname_in,
                                      grid_size=grid_size, dielec_const=dielec_const)

                    with open(apbs_in_fname, "w") as f:
                        f.write(input)

                    # generates dx.gz grid file
                    proc = subprocess.Popen(
                        ["apbs", apbs_in_fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    cmd_out = proc.communicate()
                    if proc.returncode != 0:
                        raise Exception(cmd_out[1].decode())

                os.chdir(owd)

            grid = PotGrid(dst_pdb_file, grid_fname)

        return grid

    def _genGrids(self, structure, ligand, grid_size, radius, dielec_const_list) -> (Mapping[float, PotGrid], np.ndarray):

        with self.tmp_file(f'{self.tmp_data_folder}/protein_trans.pdb') as dst_pdb_file:

            pr.writePDB(dst_pdb_file, structure)
            # pdb2pqr fails to read pdbs with the one line header generated by ProDy...
            with open(dst_pdb_file, 'r') as fin:
                data = fin.read().splitlines(True)
            with open(dst_pdb_file, 'w') as fout:
                fout.writelines(data[1:])

            grids = self._runApbs(dst_pdb_file=dst_pdb_file, grid_size=grid_size, dielec_const_list=dielec_const_list)

        # ligand mask is a boolean NumPy array, can be converted to int: ligand_mask.astype(int)
        ligand_mask = next(iter(grids.values())).get_ligand_mask(ligand, radius)

        return grids, ligand_mask