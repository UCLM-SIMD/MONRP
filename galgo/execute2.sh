#!/bin/sh

PBS -l mem=15gb

# Make sure that the script is run
# in the current working directory
cd $CWD

declare -a algorithms=("genetic","geneticnds","nsgaii")
declare -a crossovers=(0.8,0.85,0.9)
declare -a mutations=(0,0.05,0.1)
declare -a replacements=("elitism","elitismNDS")
declare -a popsize=(20,30,40)
declare -a generations=(100,200,300)
declare -a datasets=("dataset1","dataset2")
declare -i seed=10
FILE_PATH="output/resultados.txt"

source activate prueba

python "$FILE" --configuration $CONFIG
${datasets[0]} seed ${algorithms[0]} ${popsize[0]} ${generations[0]} ${crossovers[0]} ${mutations[0]} ${replacements[0]} FILE_PATH
