from dataset1 import generate_dataset1_genes
from executer import executer, initialize_file
from algorithms.genetic.genetic_algorithm import GeneticAlgorithm
from algorithms.genetic_nds.genetic_nds_algorithm import GeneticNDSAlgorithm
from algorithms.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
from models.problem import Problem

# generar los genes de los requisitos------------------------------------------------------------------
genes=generate_dataset1_genes()

# score=MAX; cost=MIN;
objectives_minimization=["MAX","MIN"]

# instanciar el problema------------------------------------------------------------------
problem=Problem(genes,objectives_minimization)

# crear seed random------------------------------------------------------------------
seed=10
seed=None

# iniciar------------------------------------------------------------------
'''
algorithm=GeneticAlgorithm(problem,random_seed=seed,population_length=20,max_generations=100,crossover_prob=0.9,mutation_prob=0.1)
executer(algorithm,iterations=5)


algorithm=GeneticNDSAlgorithm(problem,random_seed=seed,population_length=20,max_generations=100,crossover_prob=0.9,mutation_prob=0.1)
executer(algorithm,iterations=5)


algorithm=NSGAIIAlgorithm(problem,random_seed=seed,population_length=20,max_generations=100,crossover_prob=0.8,mutation_prob=0.05)
executer(algorithm,iterations=40)
'''

cross=[0.8,0.85,0.9]
mut=[0,0.05,0.1]
pop=[20,30,40]
gen=[100,200,300]
FILE_PATH="output/executer_total.txt"
initialize_file(FILE_PATH)
for c in cross:
	for m in mut:
		for p in pop:
			for g in gen:
				algorithm = NSGAIIAlgorithm(problem, random_seed=seed, population_length=p, max_generations=g,
											crossover_prob=c, mutation_prob=m)
				executer(algorithm, iterations=10,file_path=FILE_PATH)
				print("next")
print("end of executions")