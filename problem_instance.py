from dataset1 import generate_dataset1_genes
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
from problem import Problem
import matplotlib.pyplot as plt

# generar los genes de los requisitos------------------------------------------------------------------
genes=generate_dataset1_genes()
# score=MAX; cost=MIN;
objectives_minimization=["MAX","MIN"]
# instanciar el problema------------------------------------------------------------------
problem=Problem(genes,objectives_minimization)

'''
algorithm=GeneticAlgorithm(problem,population_length=20,max_evaluations=5000,crossover_prob=0.9,mutation_prob=0.1,)
algorithm.run()
print("Best individual: ",algorithm.best_individual)
'''

algorithm=NSGAIIAlgorithm(problem,population_length=20,max_generations=1000,crossover_prob=0.9,mutation_prob=0.1)
front=algorithm.run()

print("Front 0:")
for ind in front:
	print(ind)

func = [i.objectives for i in front]
function1 = [i[0].value for i in func]
function2 = [i[1].value for i in func]
plt.xlabel('Function score [MAX]', fontsize=15)
plt.ylabel('Function cost [MIN]', fontsize=15)
plt.scatter(function1, function2)
plt.show()
