#!/bin/bash

SERVER=codon-login
DIR=/hps/nobackup/arl/chembl/lorenzo/3dunet-cavity/runs/run_210601/predictions/
rsync -avz -e ssh ${SERVER}:${DIR} .
