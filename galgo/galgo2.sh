#!/bin/bash

source activate prueba

MODULE="$1"
MODULE_NAME=$(echo $MODULE | tr [a-z] [A-Z])

# Execute one job array per dataset
DATASETS=$(python $PWD/datasets.py)

for DATASET in $DATASETS
do
    # The name of the job combine
    # the module name and dataset
    NAME="$MODULE_NAME"_"$DATASET"
    NAME=$(echo $NAME | tr [a-z] [A-Z])

    # Get the number of jobs
    JOBS=$(python $PWD/$MODULE.py)

    qsub -J 1-"$JOBS" \
         -N "$NAME" \
         -e "$PWD"/errors/ \
         -o "$PWD"/outputs/ \
         -v FILE="$PWD"/main.py,MODULE="$MODULE",DATASET="$DATASET",CWD="$PWD" \
          "$PWD"/execute.sh
done
