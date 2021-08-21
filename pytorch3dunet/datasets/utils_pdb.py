import os
import shutil
from pathlib import Path
from openbabel import openbabel
import prody as pr
from pytorch3dunet.unet3d.utils import get_logger

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