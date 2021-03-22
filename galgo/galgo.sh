#!/bin/bash
source ../venv/bin/activate
# Execute one job array per dataset
python $PWD/galgo/configurations.py

CONFIGURATIONS=()
while IFS= read -r line; do
   CONFIGURATIONS+=("$line")
done <$PWD/configs.txt
#i=0
#while read line
#do
 #       arr[$i]="$line"
  #      i=$((i+1))
 #   echo "$line"
#done < config.txt

#for CONFIG in $CONFIGURATIONS
#do
    # The name of the job combine
    # the module name and dataset
    NAME="monrp"
    #NAME=$(echo $NAME | tr [a-z] [A-Z])

    # Get the number of jobs
    #JOBS=$(python $PWD/$MODULE.py)
    JOBS=${#CONFIGURATIONS[@]}
    echo "$JOBS"

    qsub -J 1-"$JOBS" \
         -N "$NAME" \
         -e "$PWD"/errors/ \
         -o "$PWD"/outputs/ \
         -v FILE="$PWD"/ejecutor_galgo.py,CWD="$PWD" "$PWD"/execute.sh
         #CONFIG="$CONFIG", \


#done
