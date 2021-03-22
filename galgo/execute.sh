#!/bin/sh

PBS -l mem=15gb

# Make sure that the script is run
# in the current working directory
cd $CWD

source activate monrp

python "$FILE" --combination $((PBS_ARRAY_INDEX - 1)) \
               --module "$MODULE" \
               --dataset "$DATASET"
