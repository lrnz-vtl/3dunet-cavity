#!/bin/bash

SERVER=codon-login
DIR=/hps/nobackup/arl/chembl/lorenzo/3dunet-cavity/runs
rsync -avz --exclude "tmp" --exclude "tmp_predict" --include "*/" --include "*.yml"  --include "*.clstr" --include "logs/*"  --exclude '*' -e ssh ${SERVER}:${DIR} .
#--include "predictions/*"
#--include "*best_checkpoint.pytorch"
