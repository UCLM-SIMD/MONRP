import os
import sys
import argparse
from algorithms.EDA.FEDA.feda_algorithm import FEDAAlgorithm
from algorithms.EDA.FEDA.feda_executer import FEDAExecuter
from algorithms.EDA.bivariate.MIMIC.mimic_algorithm import MIMICAlgorithm
from algorithms.EDA.bivariate.MIMIC.mimic_executer import MIMICExecuter
sys.path.append(os.getcwd()) 
from algorithms.EDA.PBIL.pbil_executer import PBILExecuter
from algorithms.EDA.PBIL.pbil_algorithm import PBILAlgorithm

from algorithms.EDA.UMDA.umda_executer import UMDAExecuter
from algorithms.EDA.UMDA.umda_algorithm import UMDAAlgorithm

from algorithms.genetic.abstract_genetic.genetic_executer import GeneticExecuter
from algorithms.genetic.abstract_genetic.abstract_genetic_algorithm import AbstractGeneticAlgorithm

from algorithms.GRASP.grasp_executer import GRASPExecuter
from algorithms.GRASP.GRASP import GRASP


curpath = os.path.abspath(os.curdir)

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm',help='<Required> algorithm', required=True)
alg = parser.parse_args().algorithm

out_dir="output/metrics"
out_file=f"merged_output_{alg}.txt"

filenames = [f for f in os.listdir(
    out_dir) if f.endswith('.txt') and alg in f]
print(filenames)

if alg=="pbil":
    PBILExecuter(PBILAlgorithm).initialize_file(
        f'{out_dir}/{out_file}')
elif alg=="umda":
    UMDAExecuter(UMDAAlgorithm).initialize_file(
        f'{out_dir}/{out_file}')
elif alg=="genetic":
    GeneticExecuter(AbstractGeneticAlgorithm).initialize_file(
        f'{out_dir}/{out_file}')
elif alg=="grasp":
    GRASPExecuter(GRASP).initialize_file(
        f'{out_dir}/{out_file}')
elif alg=="mimic":
    MIMICExecuter(MIMICAlgorithm).initialize_file(
        f'{out_dir}/{out_file}')
elif alg=="feda":
    FEDAExecuter(FEDAAlgorithm).initialize_file(
        f'{out_dir}/{out_file}')
else:
    raise Exception("Error, algorithm not found")
    
with open(f'{out_dir}/{out_file}', 'a') as outfile:
    for names in filenames:
        with open(f'{out_dir}/{names}') as infile:
            outfile.write(infile.read())
    outfile.close()
