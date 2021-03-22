#!/bin/sh

#PBS -l mem=15gb

# Make sure that the script is run
# in the current working directory
cd $CWD

source ../venv/bin/activate

CONFIGURATIONS=()
while IFS= read -r line; do
   CONFIGURATIONS+=("$line")
done <$PWD/configs.txt
#echo "aaa"
#echo "${PBS_ARRAY_INDEX}"
#echo "$(PBS_ARRAY_INDEX-1)"
#echo "${CONFIGURATIONS[$(PBS_ARRAY_INDEX-1)]}"

#echo "${CONFIGURATIONS[$(PBS_ARRAY_INDEX-1)]}"
#echo "$((PBS_ARRAY_INDEX - 1))"
CONFIG="${CONFIGURATIONS[$((PBS_ARRAY_INDEX - 1))]}"
#echo "$CONFIG"
#python "$FILE" -c "$CONFIG" #--combination $((PBS_ARRAY_INDEX - 1)) \
python "$FILE" -c "$CONFIG"