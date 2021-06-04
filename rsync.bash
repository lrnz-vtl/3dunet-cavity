#!/bin/bash

SERVER=codon-login
DIR=/hps/nobackup/arl/chembl/lorenzo/3dunet-cavity/runs
rsync -avz --include "*/" --include "*.yml" --include "*best_checkpoint.pytorch" --include "*.clstr" --include "logs/*" --include "predictions/*" --exclude '*' -e ssh ${SERVER}:${DIR} .
