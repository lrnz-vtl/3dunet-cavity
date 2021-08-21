import os
from pytorch3dunet.unet3d import utils
from argparse import ArgumentParser
import yaml
from pathlib import Path
from pytorch3dunet.datasets.utils_pdb import processPdb
import prody as pr
from shutil import copyfile

logger = utils.get_logger('DataGen')

def procName(arg):
    output, name, dataFolder = arg

    output_dir = str(Path(output) / name)
    os.makedirs(output_dir, exist_ok=True)

    pdb_ligand_fname = str(Path(output_dir) / f"{name}_ligand.pdb")
    structure, ligand = processPdb(dataFolder, name, pdb_ligand_fname)

    src_mol_file = f"{dataFolder}/{name}/{name}_ligand.mol2"
    dst_pdb_file = f'{output_dir}/{name}_selected.pdb'
    dst_mol_file = f'{output_dir}/{name}_ligand.mol2'
    pr.writePDB(dst_pdb_file, structure)

    copyfile(src_mol_file, dst_mol_file)

    # Generate ground truth pocket
    dst_pocket_file = f'{output_dir}/{name}_pocket.pdb'
    structure = pr.parsePDB(dst_pdb_file)
    complx = ligand + structure
    lresname = ligand.getResnames()[0]
    pocket = complx.select(f'same residue as exwithin 4.5 of resname {lresname}')
    pr.writePDB(dst_pocket_file, pocket)


def main():
    parser = ArgumentParser()
    parser.add_argument("-r", "--runconfig", dest='runconfig', type=str, required=True,
                        help=f"The run config yaml file")
    parser.add_argument("-o", "--output", dest='output', type=str, required=True,
                        help=f"Output folder")

    args = parser.parse_args()
    runconfigPath = args.runconfig

    runconfig = yaml.safe_load(open(runconfigPath, 'r'))
    dataFolder = Path(runconfig['dataFolder'])

    os.makedirs(args.output, exist_ok=True)
    logger.info(f'Dumping data to: {args.output}')

    for name in runconfig['test']:
        arg = args.output, name, dataFolder
        procName(arg)

if __name__ == '__main__':
    main()