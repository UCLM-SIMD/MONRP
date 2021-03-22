from dataset1 import generate_dataset1_genes
from dataset2 import generate_dataset2_genes
from executer import executer, initialize_file
from algorithms.genetic.genetic_algorithm import GeneticAlgorithm
from algorithms.genetic_nds.genetic_nds_algorithm import GeneticNDSAlgorithm
from algorithms.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
from models.problem import Problem
from dask import delayed as delay
import time
# iniciar------------------------------------------------------------------
'''
algorithm=GeneticAlgorithm(problem,random_seed=seed,population_length=20,max_generations=100,crossover_prob=0.9,mutation_prob=0.1)
executer(algorithm,iterations=5)
algorithm=GeneticNDSAlgorithm(problem,random_seed=seed,population_length=20,max_generations=100,crossover_prob=0.9,mutation_prob=0.1)
executer(algorithm,iterations=5)
algorithm=NSGAIIAlgorithm(problem,random_seed=seed,population_length=20,max_generations=100,crossover_prob=0.8,mutation_prob=0.05)
executer(algorithm,iterations=40)
'''
#@delay
def loop_executions():
	counter = 0
	max_counter = len(cross) * len(mut) * len(pop) * len(generations)  * len(algorithms) * len(dataset_problems)
	for c in cross:
		for m in mut:
			for p in pop:
				for g in generations:
					for dataset_problem in dataset_problems:
						for selected_algorithm in algorithms:
							start = time.time()
							algorithm = selected_algorithm(dataset_problem["problem"], random_seed=seed, population_length=p, max_generations=g,crossover_prob=c, mutation_prob=m,
														   replacement = "elitism")
							executer(algorithm,dataset=dataset_problem["name"], iterations=1, file_path=FILE_PATH)

							counter += 1
							end = time.time()
							print("Percentage: ", (counter / max_counter) * 100, " %.   Time elapsed: ", end - start)

	print("End of executions-------------------------")



# generar los genes de los requisitos------------------------------------------------------------------
genes=generate_dataset1_genes()
genes2=generate_dataset2_genes()
# score=MAX; cost=MIN;
objectives_minimization=["MAX","MIN"]

# instanciar el problema------------------------------------------------------------------
problem1=Problem(genes,objectives_minimization)
problem2=Problem(genes2,objectives_minimization)

# crear seed random------------------------------------------------------------------
seed=10
# opciones------------------------------------------------------------------
cross=[0.8
	#,0.85,0.9
	   ]
mut=[0
	#,0.05,0.1
	 ]
pop=[20
	#,30,40
	 ]
generations=[100
	#,200,300
	]
dataset_problems=[
	{"problem":problem1,
	 "name":"dataset1"
	 },
	#{"problem":problem2,
	# "name":"dataset2"}
]
algorithms=[
	GeneticAlgorithm,
	GeneticNDSAlgorithm,
	NSGAIIAlgorithm
]

FILE_PATH="output/resultados.txt"
initialize_file(FILE_PATH)

loop_executions()#.compute()
