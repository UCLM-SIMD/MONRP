#!/bin/bash

source activate prueba

# Execute one job array per dataset
CONFIGURATIONS=$(python $PWD/galgo/configurations.py)

for CONFIG in $CONFIGURATIONS
do
    # The name of the job combine
    # the module name and dataset
    NAME="MONRP"_"$CONFIG"
    NAME=$(echo $NAME | tr [a-z] [A-Z])

    # Get the number of jobs
    #JOBS=$(python $PWD/$MODULE.py)
    JOBS=10

    qsub -J 1-"$JOBS" \
         -N "$NAME" \
         -e "$PWD"/errors/ \
         -o "$PWD"/outputs/ \
         -v FILE="$PWD"/ejecutor_galgo.py,CONFIG="$CONFIG",CWD="$PWD", \
          "$PWD"/execute2.sh
done
