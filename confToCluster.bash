#!/bin/bash

SRC=/home/lorenzo/3dunet-cavity/runs/"$1"
SERVER=codon-login
# DST=/hps/nobackup/arl/chembl/lorenzo/3dunet-cavity/runs
DST=/homes/vitale/tmp

if [ "$1" = "" ] ;
# || [ $# -gt 1 ]; 
then
	echo "Usage: ${0} <RUN NAME> [RSYNC ARGS]"
else
	echo "Copying ${SRC}"
	rsync -avz --include "*/" --include "*.yml" --include "*.clstr" --exclude '*' -e ssh ${SRC} ${SERVER}:${DST} "${@:2}"
fi



