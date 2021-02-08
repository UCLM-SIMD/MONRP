from dataset1 import generate_dataset1_genes
from executer import executer
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

algorithm=GeneticAlgorithm(problem,random_seed=seed,population_length=20,max_generations=100,crossover_prob=0.9,mutation_prob=0.1)
executer(algorithm,iterations=5)


algorithm=GeneticNDSAlgorithm(problem,random_seed=seed,population_length=20,max_generations=100,crossover_prob=0.9,mutation_prob=0.1)
executer(algorithm,iterations=5)


algorithm=NSGAIIAlgorithm(problem,random_seed=seed,population_length=20,max_generations=100,crossover_prob=0.9,mutation_prob=0.5)
executer(algorithm,iterations=5)


