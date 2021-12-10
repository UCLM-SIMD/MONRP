#!/bin/sh
# 8gb of memory should be enough

#PBS -l mem=8gb

# Make sure that the script is run
# in the current working directory
cd $CWD

source ../newvenv/bin/activate

python "$PWD"/backtracking_algorithm.py -d 2