#!/bin/sh

PBS -l mem=15gb

# Make sure that the script is run
# in the current working directory
cd $CWD

source activate prueba

python "$FILE" --configuration $CONFIG
