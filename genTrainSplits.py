import os
import yaml
import glob
from pathlib import Path
from argparse import ArgumentParser
import sys
import shutil
import re
import random

outputName = "run_config.yml"

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("-b", "--data", dest='base_data', type=str, required=True,
                        help="base folder where to read the .h5 datasets")
    parser.add_argument("-d", "--dst", dest='dest', type=str, required=True,
                        help="destination folder")
    parser.add_argument("-c", "--clusterfile", dest='clusterfile', type=str, default=None,
                        help="Optional, the output of cd-hit used to determine clusters (.clstr)")

    args = parser.parse_args()

    base_data = Path(os.path.abspath(args.base_data))
    base_run = Path(os.path.abspath(args.dest))

    clusterfname = args.clusterfile
    if clusterfname is not None and not os.path.exists(clusterfname):
        print(f"Cluster file '{clusterfname}' not found", file=sys.stderr)
        sys.exit(-1)

    if not os.path.exists(args.base_data):
        print(f"Folder '{args.base_data}' not found", file=sys.stderr)
        sys.exit(-1)

    names_train = []
    names_val = []
    names_test = []

    if clusterfname is None:
        print("Randomising splits. No clusters are considered.")
        useClusters = False

        names = []
        for ddir in glob.glob(str(base_data) + rf"\**\*_grids.h5"):
            pdir = Path(ddir)
            names.append(pdir.parent.name)

        print(f"Found {len(names)} data files.")

        n = len(names)
        ntrain = int(n * 0.6)
        nval = int(n * 0.2)

        random.seed(0)
        random.shuffle(names)

        names_train = names[:ntrain]
        names_val = names[ntrain:ntrain + nval]
        names_test = names[ntrain + nval:]
    else:
        print("Reading clusters from file.")

        clusters = []
        with open(clusterfname) as f:
            cluster = []
            for line in f.readlines():
                print(line)
                if re.match(r'>Cluster', line) is not None:
                    clusters.append(cluster)
                    cluster = []
                else:
                    m = re.search(r'>(.*?)\|', line)
                    if m is None:
                        print(f"Found unparsable line: {line}", file=sys.stderr)
                        sys.exit(-1)
                    cluster.append(m.group(1))
        print(clusters)
        random.seed(0)
        random.shuffle(clusters)

        i = 0
        n = sum([len(c) for c in clusters])
        while len(names_train) < n*0.6:
            names_train += clusters[i]
            i += 1
        while len(names_val) < n*0.2:
            names_val += clusters[i]
            i += 1
        while i < len(clusters):
            names_test += clusters[i]
            i += 1

        useClusters = True
        dstclusterfile = base_run / Path(clusterfname).name
        shutil.copy(clusterfname, dstclusterfile)


    x = {'train': names_train, 'val': names_val, 'test': names_test,
         'runFolder': str(base_run),
         'dataFolder': str(base_data),
         'useClusters' : useClusters}

    os.makedirs(base_run, exist_ok=True)
    dstFname = base_run / outputName
    with open(dstFname, 'w') as f:
        yaml.dump(x, f, allow_unicode=True)

    print(f"Written config file to {dstFname}")