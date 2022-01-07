# RUN SCRIPT IN THE FOLDER WHERE THE DATA IS LOCATED
import os
import sys
sys.path.append(os.getcwd()) 
from algorithms.EDA.UMDA.umda_executer import UMDAExecuter
from algorithms.EDA.UMDA.umda_algorithm import UMDAAlgorithm

out_dir="output/metrics"
out_file="merged_output_umda.txt"
filenames = [f for f in os.listdir(
    out_dir) if f.endswith('.txt') and "umda" in f]
print(filenames)

UMDAExecuter(UMDAAlgorithm).initialize_file(
    f'{out_dir}/{out_file}')

with open(f'{out_dir}/{out_file}', 'a') as outfile:
    for names in filenames:
        with open(f'{out_dir}/{names}') as infile:
            outfile.write(infile.read())
    outfile.close()
