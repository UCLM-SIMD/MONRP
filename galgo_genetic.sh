#!/bin/bash
# execute galgo.sh from project directory
# project directory files: galgo.sh, execute.sh, galgo/configurations.py, configs.txt (autogenerated)
source ../newvenv/bin/activate
# Execute one job array per dataset
python $PWD/galgo/genetic/configurations_genetic.py

CONFIGURATIONS=()
while IFS= read -r line; do
   CONFIGURATIONS+=("$line")
done <$PWD/configs_genetic.txt

NAME="monrp-genetic"
JOBS=${#CONFIGURATIONS[@]}
echo "$JOBS"

qsub -J 1-"$JOBS" \
     -N "$NAME" \
     -e "$PWD"/errors/ \
     -o "$PWD"/outputs/ \
     -v FILE="$PWD"/executer_driver.py,CWD="$PWD",CONFIG_FILE=configs_genetic.txt "$PWD"/execute.sh

