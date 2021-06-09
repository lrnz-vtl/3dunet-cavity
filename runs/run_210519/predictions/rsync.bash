#!/bin/bash

SERVER=codon-login
DIR=/hps/nobackup/arl/chembl/lorenzo/3dunet-cavity/runs/run_210519/predictions/
rsync -avz -e ssh ${SERVER}:${DIR} . $@
