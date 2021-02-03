from dataset1 import generate_dataset1_genes
from executer import executer
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
from problem import Problem

# generar los genes de los requisitos------------------------------------------------------------------
genes=generate_dataset1_genes()

# score=MAX; cost=MIN;
objectives_minimization=["MAX","MIN"]

# instanciar el problema------------------------------------------------------------------
problem=Problem(genes,objectives_minimization)

# crear seed random------------------------------------------------------------------
seed=10

# iniciar------------------------------------------------------------------
print("Running...")

algorithm=GeneticAlgorithm(problem,population_length=20,max_evaluations=000,crossover_prob=0.9,mutation_prob=0.1,random_seed=seed)
#algorithm=NSGAIIAlgorithm(problem,random_seed=seed,population_length=20,max_generations=100,crossover_prob=0.9,mutation_prob=0.1)

executer(algorithm,10)


