from algorithms.genetic.genetic_algorithm import GeneticAlgorithm
from algorithms.genetic_nds.genetic_nds_algorithm import GeneticNDSAlgorithm
from algorithms.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
from datasets.dataset1 import generate_dataset1_genes
from datasets.dataset2 import generate_dataset2_genes
from executer import executer, reset_file
from models.problem import Problem
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', nargs='+', help='<Required> configuration', required=True)
# PARAMETROS: dataset_name, seed, algoritmo, pop_length, max_gens, sel_scheme, cross_scheme, cross_prob, mut_scheme, mut_prob, repl_scheme

for _, value in parser.parse_args()._get_kwargs():
	if value is not None:
		parameters = str(value[0])
		parameters = parameters.split()
		dataset_name = str(parameters[0])
		seed = int(parameters[1])
		algorithm_name = str(parameters[2])
		pop_length = int(parameters[3])
		max_gens = int(parameters[4])
		sel_scheme = str(parameters[5])
		cross_scheme = str(parameters[6])
		cross_prob = float(parameters[7])
		mut_scheme = str(parameters[8])
		mut_prob = float(parameters[9])
		repl_scheme = str(parameters[10])  # repl = elitismNDS o elitism
		# filepath = str(value[8])


objectives_minimization = ["MAX", "MIN"]

filepath = "output/"+str(dataset_name)+"-"+str(seed)+"-"+str(algorithm_name)+"-"+str(pop_length)+"-"+str(max_gens) +"-"+str(sel_scheme) \
		   + "-" + str(cross_scheme) + "-" + str(cross_prob)+"-"+str(mut_scheme)+"-"+str(mut_prob) +"-"+str(repl_scheme)+".txt"
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

if dataset_name =="dataset1":
	genes = generate_dataset1_genes()
	problem = Problem(genes, objectives_minimization)
elif dataset_name =="dataset2":
	genes = generate_dataset2_genes()
	problem = Problem(genes, objectives_minimization)

if algorithm_name == "genetic":
	algorithm = GeneticAlgorithm
elif algorithm_name == "geneticnds":
	algorithm = GeneticNDSAlgorithm
elif algorithm_name == "nsgaii":
	algorithm = NSGAIIAlgorithm

algorithm = algorithm(problem, random_seed=seed, population_length=pop_length, max_generations=max_gens,
					selection=sel_scheme,crossover=cross_scheme, crossover_prob=cross_prob,mutation=mut_scheme,
							   mutation_prob=mut_prob,replacement = repl_scheme)
#initialize_file(filepath)
reset_file(filepath)
executer(algorithm,dataset=dataset_name, iterations=10, file_path=filepath)

