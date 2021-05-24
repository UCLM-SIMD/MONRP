from algorithms.GRASP.GRASP import GRASP
from algorithms.genetic.genetic_algorithm import GeneticAlgorithm
from algorithms.genetic_nds.genetic_nds_algorithm import GeneticNDSAlgorithm
from algorithms.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
from datasets.dataset1 import generate_dataset1_genes
from datasets.dataset2 import generate_dataset2_genes
from executer import Executer
from models.problem import Problem
import argparse
import os
curpath = os.path.abspath(os.curdir)

parser = argparse.ArgumentParser()
#parser.add_argument('-ge', '--genetic', nargs=11, help='<Required> configuration', required=False)
#parser.add_argument('-gr', '--grasp', nargs=6, help='<Required> configuration', required=False)

parser.add_argument('-c', '--config', nargs="+", help='<Required> configuration', required=False)


params = parser.parse_args().config[0].split()#sh galgo
#params = parser.parse_args().config#local
print(type(params))
print(params)
if(params[0]=="genetic"):
	dataset_name = str(params[1])
	seed = int(params[2])
	algorithm_name = str(params[3])
	pop_length = int(params[4])
	max_gens = int(params[5])
	sel_scheme = str(params[6])
	cross_scheme = str(params[7])
	cross_prob = float(params[8])
	mut_scheme = str(params[9])
	mut_prob = float(params[10])
	repl_scheme = str(params[11])
	objectives_minimization = ["MAX", "MIN"]
	filepath = "output/genetic-"+(str(dataset_name)+"-"+str(seed)+"-"+str(algorithm_name)+"-"+str(pop_length)+"-"+str(max_gens) +"-"+str(sel_scheme) \
						  + "-" + str(cross_scheme) + "-" + str(cross_prob)+"-"+str(mut_scheme)+"-"+str(mut_prob) +"-"+str(repl_scheme)+".txt")

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

	executer = Executer("genetic")

elif(params[0]=="grasp"):
	dataset_name,algorithm_name,iterations,solutions_per_iteration,local_search_type,seed = \
		[int(params[1]),str(params[2]),int(params[3]),int(params[4]),(params[5]),int(params[6])]

	filepath = "output/grasp-"+(str(dataset_name)+"-"+str(seed)+"-"+str(algorithm_name)+"-"+str(iterations)+"-"+str(solutions_per_iteration) \
						  +"-"+str(local_search_type)+".txt")

	if algorithm_name == "grasp":
		algorithm = GRASP

	algorithm = algorithm(dataset=dataset_name, iterations=iterations, solutions_per_iteration=solutions_per_iteration,
						  local_search_type=local_search_type, seed=seed)

	executer = Executer("grasp")

else:
	print(params[0])
# PARAMETROS: dataset_name, seed, algoritmo, pop_length, max_gens, sel_scheme, cross_scheme, cross_prob, mut_scheme, mut_prob, repl_scheme
'''
genetic_params = parser.parse_args().genetic
grasp_params = parser.parse_args().grasp
if genetic_params is not None and len(genetic_params)>0:
	dataset_name = str(genetic_params[0])
	seed = int(genetic_params[1])
	algorithm_name = str(genetic_params[2])
	pop_length = int(genetic_params[3])
	max_gens = int(genetic_params[4])
	sel_scheme = str(genetic_params[5])
	cross_scheme = str(genetic_params[6])
	cross_prob = float(genetic_params[7])
	mut_scheme = str(genetic_params[8])
	mut_prob = float(genetic_params[9])
	repl_scheme = str(genetic_params[10])  # repl = elitismNDS o elitism
	# filepath = str(value[8])

	objectives_minimization = ["MAX", "MIN"]
	filepath = "output/"+(str(dataset_name)+"-"+str(seed)+"-"+str(algorithm_name)+"-"+str(pop_length)+"-"+str(max_gens) +"-"+str(sel_scheme) \
		   + "-" + str(cross_scheme) + "-" + str(cross_prob)+"-"+str(mut_scheme)+"-"+str(mut_prob) +"-"+str(repl_scheme)+".txt")

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

	executer = Executer("genetic")

elif grasp_params is not None and len(grasp_params)>0:
	dataset_name,algorithm_name,iterations,solutions_per_iteration,local_search_type,seed = \
		[int(grasp_params[0]),str(grasp_params[1]),int(grasp_params[2]),int(grasp_params[3]),(grasp_params[4]),int(grasp_params[5])]
	filepath = "output/"+(str(dataset_name)+"-"+str(seed)+"-"+str(algorithm_name)+"-"+str(iterations)+"-"+str(solutions_per_iteration) \
							+"-"+str(local_search_type)+".txt")

	if algorithm_name == "grasp":
		algorithm = GRASP

	algorithm = algorithm(dataset=dataset_name, iterations=iterations, solutions_per_iteration=solutions_per_iteration,
						  local_search_type=local_search_type, seed=seed)

	executer = Executer("grasp")
'''




'''
print(dataset_name)
print(seed)
print(algorithm_name)
print(pop_length)
print(max_gens)
print(cross_prob)
print(mut_prob)
print(repl_scheme)
print(filepath)
'''

#if not os.path.isfile(filepath):
#	executer.initialize_file(filepath)
#filepath = os.path.join(curpath, filepath)
#print(filepath)
#executer.reset_file(filepath)
executer.execute(algorithm,dataset=dataset_name, executions=10, file_path=filepath)

