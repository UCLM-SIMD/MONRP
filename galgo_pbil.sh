#!/bin/bash
# execute galgo.sh from project directory
# project directory files: galgo.sh, execute.sh, galgo/configurations.py, configs.txt (autogenerated)
source ../newvenv/bin/activate
# Execute one job array per dataset
python $PWD/galgo/eda/pbil/configurations_pbil.py

CONFIGURATIONS=()
while IFS= read -r line; do
   CONFIGURATIONS+=("$line")
done <$PWD/configs_pbil.txt

NAME="monrp-pbil"
JOBS=${#CONFIGURATIONS[@]}
echo "$JOBS"

qsub -J 1-"$JOBS" \
     -N "$NAME" \
     -e "$PWD"/errors/ \
     -o "$PWD"/outputs/ \
     -v FILE="$PWD"/executer_driver.py,CWD="$PWD",CONFIG_FILE=configs_pbil.txt "$PWD"/execute.sh
