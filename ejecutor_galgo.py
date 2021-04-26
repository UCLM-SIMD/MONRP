import time
import sys
from algorithms.genetic.genetic_algorithm import GeneticAlgorithm
from algorithms.genetic_nds.genetic_nds_algorithm import GeneticNDSAlgorithm
from algorithms.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
from dataset1 import generate_dataset1_genes
from dataset2 import generate_dataset2_genes
from executer import executer
from models.problem import Problem
import argparse

print("Starting...")
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', nargs='+', help='<Required> configuration', required=True)
# PARAMETROS: dataset_name, seed, algoritmo, pop_length, max_gens, cross_prob, mut_prob, replace, filepath

for _, value in parser.parse_args()._get_kwargs():
	if value is not None:
		dataset_name = str(value[0])
		seed = int(value[1])
		algorithm_name = str(value[2])
		pop_length = int(value[3])
		max_gens = int(value[4])
		cross_prob = float(value[5])
		mut_prob = float(value[6])
		repl = str(value[7])  # repl = elitismNDS o elitism
		#filepath = str(value[8])
		objectives_minimization = ["MAX", "MIN"]

filepath = "/home/pbermejo/MONRP/output/"+str(dataset_name)+"-"+str(seed)+"-"+str(algorithm_name)+"-"+str(pop_length)+"-"+str(max_gens)+"-"+str(cross_prob)\
		   +"-"+str(mut_prob)+"-"+str(repl)+".txt"

'''
print(dataset_name)
print(seed)
print(algorithm_name)
print(pop_length)
print(max_gens)
print(cross_prob)
print(mut_prob)
print(repl)
print(filepath)
'''


execute = True
if dataset_name =="dataset1":
	genes = generate_dataset1_genes()
	problem = Problem(genes, objectives_minimization)
elif dataset_name =="dataset2":
	genes = generate_dataset2_genes()
	problem = Problem(genes, objectives_minimization)

if algorithm_name == "genetic":
	algorithm = GeneticAlgorithm
	if repl != "elitism":
		execute = False
elif algorithm_name == "geneticnds":
	algorithm = GeneticNDSAlgorithm
elif algorithm_name == "nsgaii":
	algorithm = NSGAIIAlgorithm
	if repl != "elitism":
		execute = False

if execute:
	algorithm = algorithm(problem, random_seed=seed, population_length=pop_length, max_generations=max_gens,
						  crossover_prob=cross_prob, mutation_prob=mut_prob,
								   replacement = repl)

	executer(algorithm,dataset=dataset_name, iterations=10, file_path=filepath)

