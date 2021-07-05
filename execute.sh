#!/bin/sh
# 8gb of memory should be enough

#PBS -l mem=8gb

# Make sure that the script is run
# in the current working directory
cd $CWD

source ../newvenv/bin/activate

CONFIGURATIONS=()
while IFS= read -r line; do
   CONFIGURATIONS+=("$line")
done <$PWD/$CONFIG_FILE
#done <$PWD/configs.txt

# get configuration at job index (1..JOBS_NUM)
CONFIG="${CONFIGURATIONS[$((PBS_ARRAY_INDEX - 1))]}"

python "$FILE" -c "$CONFIG"