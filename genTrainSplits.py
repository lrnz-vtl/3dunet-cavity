import os
import yaml
import glob
from pathlib import Path
from argparse import ArgumentParser
import sys

outputName = "run_config.yml"

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("-b", "--data", dest='base_data', type=str, required=True,
                        help="base folder where to read the .h5 datasets")
    parser.add_argument("-d", "--dst", dest='dest', type=str, required=True,
                        help="destination folder")

    args = parser.parse_args()

    base_data = Path(os.path.abspath(args.base_data))
    base_run = Path(os.path.abspath(args.dest))

    if not os.path.exists(args.base_data):
        print(f"Folder '{args.base_data}' not found", file=sys.stderr)
        sys.exit(-1)

    names = []

    # glob.glob(str(base_data) + rf"\**\*_grids.h5")
    for ddir in glob.glob(str(base_data) + rf"\**\*_grids.h5"):
        pdir = Path(ddir)
        names.append(pdir.parent.name)

    print(f"Found {len(names)} data files.")

    n = len(names)
    ntrain = int(n * 0.6)
    nval = int(n * 0.2)

    names_train = names[:ntrain]
    names_val = names[ntrain:ntrain + nval]
    names_test = names[ntrain + nval:]

    x = {'train': names_train, 'test': names_val, 'val': names_test,
         'runFolder': str(base_run),
         'dataFolder': str(base_data)}

    os.makedirs(base_run, exist_ok=True)
    dstFname = base_run / outputName
    with open(dstFname, 'w') as f:
        yaml.dump(x, f, allow_unicode=True)

    print(f"Written config file to {dstFname}")