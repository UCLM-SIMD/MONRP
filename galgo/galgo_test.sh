#!/bin/bash
#eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
#source activate /home/victor.perezpiqueras/miniconda/envs/prueba
source activate prueba
#conda activate prueba
#source $HOME/miniconda/bin/activate base

NAME=prueba_monrp
CONFIG=(dataset1 10 genetic 20 300 0.8 0 elitism)
qsub -N "$NAME" \
         -e "$PWD"/errors/ \
         -o "$PWD"/outputs/ \
         -v FILE="$PWD"/ejecutor_galgo.py,CWD="$PWD", CONFIG="$CONFIG" "$PWD"/execute.sh
