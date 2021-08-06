#!/bin/bash

NAME=210805_pdb

SERVER=codon-login
# DIR=/hps/nobackup/arl/chembl/lorenzo/3dunet-cavity/runs
DIR=/homes/vitale
rsync -avz -e ssh ./runs/${NAME} ${SERVER}:${DIR}
