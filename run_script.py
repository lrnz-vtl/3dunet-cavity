from pathlib import Path
from argparse import ArgumentParser
import subprocess
import os

basepath = Path('/home/lorenzo/3dunet-cavity')


def command(runname: str):
    runconfig = basepath / runname
    logfile = runconfig.parent / 'log.txt'
    scriptname = basepath / 'train.py'
    assert os.path.exists(runconfig)
    return f'python {scriptname} -r {runconfig} --logfile {logfile}'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-r", "--runname", dest='runname', type=str, required=True)

    args, unknownargs = parser.parse_known_args()
    cmd = command(args.runname).split()
    cmd = cmd + unknownargs

    print(f'Executing: {cmd}')

    subprocess.run(cmd)


