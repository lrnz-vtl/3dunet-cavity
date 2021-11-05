from __future__ import annotations
import os
from openbabel import openbabel
import prody as pr
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.datasets.featurizer import BaseFeatureList
from pytorch3dunet.datasets.features import PotentialGrid
from pytorch3dunet.datasets.apbs import ApbsGridCollection, TmpFile
from scipy.spatial.transform import Rotation
from multiprocessing import Pool, cpu_count
import numpy as np
import importlib
from openbabel.pybel import readfile, Molecule
from typing import List, Mapping, Optional, Callable, Any, Collection
from pytorch3dunet.datasets.config import PdbDataConfig, LoadersConfig, Phase
from pathlib import Path

miniball_spec = importlib.util.find_spec("miniball")
if miniball_spec is not None:
    import miniball

logger = get_logger('UtilsPdb')


def processPdb(src_data_folder, name, pdb_ligand_fname):
    """ Generate pdb training data from raw data """

    src_pdb_file = f'{src_data_folder}/{name}/{name}_protein.pdb'
    src_mol_file = f"{src_data_folder}/{name}/{name}_ligand.mol2"

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol2", "pdb")

    # remove water molecules (found some pdb files with water molecules)
    structure = pr.parsePDB(src_pdb_file)
    protein = structure.select('protein').toAtomGroup()

    # convert ligand to pdb
    ligand = openbabel.OBMol()

    obConversion.ReadFile(ligand, src_mol_file)
    obConversion.WriteFile(ligand, pdb_ligand_fname)

    # select only chains that are close to the ligand (I love ProDy v2)
    ligand = pr.parsePDB(pdb_ligand_fname)
    lresname = ligand.getResnames()[0]
    complx = ligand + protein

    # select ONLY atoms that belong to the protein
    complx = complx.select(f'same chain as exwithin 7 of resname {lresname}')
    complx = complx.select(f'protein and not resname {lresname}')

    return complx, ligand


class PdbDataHandler:
    def __init__(self,
                 src_data_folder,
                 name,
                 tmp_data_folder,
                 pdb2pqrPath,
                 cleanup: bool,
                 reuse_grids: bool,
                 cache_folder: Optional[str],
                 pregrid_transformer_config: List[Mapping] = None,
                 generating_cache: bool = False):

        if pregrid_transformer_config is None:
            pregrid_transformer_config = []

        self.cache_folder = cache_folder
        self.src_data_folder = src_data_folder
        self.name = name
        self.tmp_data_folder = tmp_data_folder
        self.pdb2pqrPath = pdb2pqrPath
        self.cleanup = cleanup
        self.apbsGrids = None
        self.reuse_grids = reuse_grids
        self.generating_cache = generating_cache

        os.makedirs(self.cache_folder, exist_ok=True)

        # TODO refactor

        if self.cache_folder is None:
            self.structure_fname = str(Path(self.tmp_data_folder) / "structure.pdb")
            self.ligand_fname = str(Path(self.tmp_data_folder) / "ligand.pdb")
        else:
            self.structure_fname = str(Path(self.cache_folder) / "structure.pdb")
            self.ligand_fname = str(Path(self.cache_folder) / "ligand.pdb")
            assert os.path.exists(self.structure_fname) or self.generating_cache
            assert os.path.exists(self.ligand_fname) or self.generating_cache

        # NOTE Lorenzo
        if self.cache_folder is not None and not self.generating_cache:
            pass
        elif self.reuse_grids and os.path.exists(self.structure_fname) and os.path.exists(self.ligand_fname):
            pass
        else:
            structure, ligand = self._processPdb()

            for elem in pregrid_transformer_config:
                if elem['name'] == 'RandomRotate':
                    structure, ligand = self._randomRotatePdb(structure, ligand)

            pr.writePDB(self.structure_fname, structure)
            pr.writePDB(self.ligand_fname, ligand)

    def checkRotations(self):
        """
            Check if the sphere inscribed in the cube contains the whole protein. If not, grid rotations
            could be disallowed for safety
        """
        structure, _ = self.getStructureLigand()
        coords = np.array(list(structure.getCoords()))

        if miniball_spec is not None:
            mb = miniball.Miniball(coords)
            C = mb.center()
            R = np.sqrt(mb.squared_radius())

            if R > self.apbsGrids.grid_size / 2:
                logger.warn(f'{self.name} cannot fit in a ball, R = {R}, C = {C}. Rotations will be turned off')
                return False

        CubeC = np.array([(e[-1] + e[0]) / 2 for e in self.apbsGrids.edges])
        dists = np.sqrt(((coords - CubeC) ** 2).sum(axis=1))
        if max(dists) > self.apbsGrids.grid_size / 2:
            logger.warn(f'{self.name} should be centered better in the cube. Rotations will be disallowed')
            return False

        return True

    def getStructureLigand(self):
        return pr.parsePDB(self.structure_fname), pr.parsePDB(self.ligand_fname)

    def getMol(self) -> Molecule:
        return next(readfile('pdb', self.structure_fname))

    def genPocket(self):
        """
        Generate ground truth pocket
        """
        structure, ligand = self.getStructureLigand()
        complx = ligand + structure
        lresname = ligand.getResnames()[0]
        ret = complx.select(f'same residue as exwithin 4.5 of resname {lresname}')

        with self.tmp_file(f'{self.tmp_data_folder}/tmp_pocket.pdb', True) as tmp_pocket_path:
            pr.writePDB(tmp_pocket_path, ret)
            ret = pr.parsePDB(tmp_pocket_path)

        return ret

    def makePdbPrediction(self, pred, expandResidues=True):

        structure, _ = self.getStructureLigand()

        if pred.ndim == 4:
            assert len(pred) == 1
            pred = pred[0]
        if pred.shape != self.apbsGrids.shape:
            raise ValueError("pred.shape != grid.shape. Are you trying to make a pocket prediction from slices? "
                             "That is currently not supported")

        predbin = pred > 0.5
        coords = []

        warned = False

        for i, coord in enumerate(structure.getCoords()):
            x, y, z = coord
            binx = int((x - min(self.apbsGrids.edges[0])) / self.apbsGrids.delta[0])
            biny = int((y - min(self.apbsGrids.edges[1])) / self.apbsGrids.delta[1])
            binz = int((z - min(self.apbsGrids.edges[2])) / self.apbsGrids.delta[2])

            if binx >= self.apbsGrids.shape[0] or biny >= self.apbsGrids.shape[1] or binz >= self.apbsGrids.shape[2]:
                if not warned:
                    logger.warn(f"Ignoring coord {i} for {self.name} because out of bounds")
                    warned = True
                continue

            if predbin[binx, biny, binz]:
                coords.append(i)

        if len(coords) == 0:
            return pr.AtomGroup()

        atoms = structure[coords]
        if expandResidues:
            idxstr = ' '.join(map(str, atoms.getIndices()))
            ret = structure.select(f'same residue as index {idxstr}')
        else:
            ret = atoms

        if len(ret) == 0:
            return ret
        else:
            with self.tmp_file(f'{self.tmp_data_folder}/tmp_pred.pdb', True) as tmp_pred_path:
                pr.writePDB(tmp_pred_path, ret)
                return pr.parsePDB(tmp_pred_path)

    def getRawsLabels(self, features: BaseFeatureList, grid_size: int, ligand_mask_radius: float):

        structure, ligand = self.getStructureLigand()
        # A different format
        mol: Molecule = self.getMol()

        dielec_const_list = features.getDielecConstList()
        if not dielec_const_list:
            # FIXME Even if empty list, need to run APBS grid anyway, to be able to generate the labels
            dielec_const_list_for_labels = [PotentialGrid.dielec_const_default]
        else:
            dielec_const_list_for_labels = dielec_const_list

        with ApbsGridCollection(structure=structure, ligand=ligand, grid_size=grid_size,
                                ligand_mask_radius=ligand_mask_radius,
                                dielec_const_list=dielec_const_list_for_labels, tmp_data_folder=self.tmp_data_folder,
                                reuse_grids=self.reuse_grids, name=self.name, pdb2pqrPath=self.pdb2pqrPath,
                                cleanup=self.cleanup, cache_folder=self.cache_folder, generating_cache=self.generating_cache) as self.apbsGrids:

            raws = features(structure, mol, self.apbsGrids)
            labels = self.apbsGrids.labels

        return raws, labels

    def _randomRotatePdb(self, structure, ligand):
        m = Rotation.random()

        r = m.as_matrix()

        angles = m.as_euler('zxy')
        logger.debug(f'Random rotation matrix angles: {list(angles)} to {self.name}')

        coords_prot = np.einsum('ij,kj->ki', r, structure.getCoords())
        coords_ligand = np.einsum('ij,kj->ki', r, ligand.getCoords())

        coords_prot_mean = coords_prot.mean(axis=0)

        # Subtract center of mass
        coords_prot = coords_prot - coords_prot_mean
        coords_ligand = coords_ligand - coords_prot_mean

        structure2 = structure.copy()
        structure2.setCoords(coords_prot)
        ligand2 = ligand.copy()
        ligand2.setCoords(coords_ligand)
        return structure2, ligand2

    def tmp_file(self, name, force_removal: bool = False):
        return TmpFile(name, self.cleanup or force_removal)

    def _processPdb(self):
        with self.tmp_file(Path(self.tmp_data_folder) / f'ligand_tmp.pdb') as tmp_ligand_pdb_file:
            return processPdb(self.src_data_folder, self.name, tmp_ligand_pdb_file)

    @classmethod
    def map_datasets(cls, loaders_config: LoadersConfig, pdb_workers: int, features_config,
                     transformer_config, phases:Collection[Phase], f: Callable[[PdbDataHandler], Any],
                     generating_cache: bool):

        data_paths = loaders_config.data_config.data_paths
        file_paths = [d for phase in phases for d in data_paths[phase]]
        file_paths = PdbDataHandler.traverse_pdb_paths(file_paths)

        args = ((file_path, name, loaders_config, features_config, transformer_config, generating_cache)
                for file_path, name in file_paths)

        if pdb_workers > 0:
            logger.info(f'Parallelizing dataset creation among {pdb_workers} workers')
            pool = Pool(processes=pdb_workers)
            return [f(x) for x in pool.map(create_dataset, args) if x is not None]
        else:
            return [f(x) for x in (create_dataset(arg) for arg in args) if x is not None]

    @staticmethod
    def traverse_pdb_paths(file_paths):

        assert isinstance(file_paths, list)
        results = []

        for file_path in file_paths:
            file_path = Path(file_path)
            name = str(file_path.name)

            pdbfile = file_path / f"{name}_protein.pdb"
            molfile = file_path / f"{name}_ligand.mol2"

            if os.path.exists(pdbfile) and os.path.exists(molfile):
                results.append((file_path.parent, name))

        return results


def create_dataset(arg) -> Optional[PdbDataHandler]:
    file_path, name, loaders_config, feature_config, transformer_config, generating_cache = arg

    loaders_config: LoadersConfig = loaders_config
    fail_on_error = loaders_config.fail_on_error

    pdb_config: PdbDataConfig = loaders_config.data_config

    cache_folder = f'{pdb_config.gridscache}/{name}'

    try:
        logger.info(f'Loading set from: {file_path} named {name} ...')
        return PdbDataHandler(
            src_data_folder=file_path,
            name=name,
            tmp_data_folder=cache_folder,
            pdb2pqrPath=pdb_config.pdb2pqrPath,
            cleanup=loaders_config.cleanup,
            reuse_grids=pdb_config.reuse_grids,
            cache_folder=cache_folder,
            pregrid_transformer_config=None,
            generating_cache=generating_cache)

    except Exception as e:
        if fail_on_error:
            raise e
        logger.error(f'Generation of grids from {name} failed. Recording failure and continuing', exc_info=True)
        Path(f'{cache_folder}/failed').touch()
        return None