#!/bin/bash

if [ $# -eq 0 ]
	then
		echo "Need to provid run name"
		exit -1
fi


SERVER=codon-login
DIR=/hps/nobackup/arl/chembl/lorenzo/3dunet-cavity/runs/${1}/predictions
rsync -avz -e ssh ${SERVER}:${DIR} ./runs/${1}
