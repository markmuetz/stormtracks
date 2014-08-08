#!/bin/sh
# Simple script to download and unzip all data from UCL computers.
# Arg 1 should be the name of the directory to get data from.
DATA_DIR=${1%%/} # Strips trailing /
cd $DATA_DIR

scp -r ucfamue@shankly.geog.ucl.ac.uk:/home/ucfamue/DATA2/stormtracks_data/output/$DATA_DIR/*.bz2 .

for Z in *.bz2; do 
    tar xvf $Z;
done

cd ..
