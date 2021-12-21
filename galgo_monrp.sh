#!/bin/bash
# execute galgo.sh from project directory
# project directory files: galgo.sh, execute.sh, galgo/configurations.py, configs.txt (autogenerated)
source ../newvenv/bin/activate

if [[ "$1" == "metrics" ]];
then
   DRIVER="executer_driver.py"
   MODE='(m)'
elif [[ "$1" == "pareto" ]];
then
   DRIVER="executer_driver_pareto.py"
   MODE='(p)'
else
   echo "[ERROR]: Allowed modes: metrics, pareto."
   echo "Example: sh galgo_monrp.sh metrics genetic"
   exit 1
fi

# Execute one job array per dataset

if [ "$2" = 'genetic' ] 
then
   python $PWD/galgo/genetic/configurations_genetic.py
   NAME="monrp-genetic${MODE}"
   FILE="configs_genetic.txt"
elif [ "$2" = 'grasp' ] 
then
   python $PWD/galgo/grasp/configurations_grasp.py
   NAME="monrp-grasp${MODE}"
   FILE="configs_grasp.txt"
elif [ "$2" = 'umda' ] 
then
   python $PWD/galgo/eda/umda/configurations_umda.py
   NAME="monrp-umda${MODE}"
   FILE="configs_umda.txt"
elif [ "$2" = 'pbil' ] 
then
   python $PWD/galgo/eda/pbil/configurations_pbil.py
   NAME="monrp-pbil${MODE}"
   FILE="configs_pbil.txt"
else
   echo "[ERROR]: Allowed algorithms: genetic, grasp, umda, pbil."
   echo "Example: sh galgo_monrp.sh metrics genetic"
   exit 1
fi

CONFIGURATIONS=()
while IFS= read -r line; do
   CONFIGURATIONS+=("$line")
done <$PWD/"$FILE"

JOBS=${#CONFIGURATIONS[@]}

echo "------Executing mode: <$1> for algorithm: <$2>------"
echo "Reading algorithm configuration from: $PWD/$FILE"
echo "Job array name: $NAME"
echo "Jobs created: $JOBS"
echo "Job array ID:"

qsub -J 1-"$JOBS" \
   -N "$NAME" \
   -e "$PWD"/errors/ \
   -o "$PWD"/outputs/ \
   -v FILE="$PWD"/"$DRIVER",CWD="$PWD",CONFIG_FILE="$FILE" "$PWD"/execute.sh