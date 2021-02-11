from algorithms.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
from dataset1 import generate_dataset1_genes
from algorithms.genetic_nds.genetic_nds_algorithm import GeneticNDSAlgorithm
from models.problem import Problem
import matplotlib.pyplot as plt

# generar los genes de los requisitos------------------------------------------------------------------
genes=generate_dataset1_genes()
# score=MAX; cost=MIN;
objectives_minimization=["MAX","MIN"]

# instanciar el problema------------------------------------------------------------------
problem=Problem(genes,objectives_minimization)

# crear seed random------------------------------------------------------------------
seed=10
#random.seed(10)
#print(random.random())
#print(random.random())

# iniciar------------------------------------------------------------------
print("Running...")

'''
algorithm=GeneticAlgorithm(problem,population_length=20,max_evaluations=000,crossover_prob=0.9,mutation_prob=0.1,random_seed=seed)
#algorithm.aa()
result=algorithm.run()
print("Best individual: ",algorithm.best_individual)

'''
'''
algorithm=GeneticNDSAlgorithm(problem,random_seed=seed,population_length=20,max_generations=200,crossover_prob=0.9,mutation_prob=0.5)
result=algorithm.run()

print("Time: ",result["time"])
print("HV: ",result["hv"])
print("Spread: ",result["spread"])
print(len(result["nds"]))
func = [i.objectives for i in result["nds"]]
function1 = [i[0].value for i in func]
function2 = [i[1].value for i in func]
plt.xlabel('Function score [MAX]', fontsize=15)
plt.ylabel('Function cost (SP) [MIN]', fontsize=15)
plt.scatter(function1, function2)
plt.show()

'''
algorithm=NSGAIIAlgorithm(problem,random_seed=seed,population_length=20,max_generations=100,crossover_prob=0.9,mutation_prob=0.5)
result=algorithm.run()

print("Time: ",result["time"])
print("HV: ",result["hv"])
print("Spread: ",result["spread"])

print("Front 0:")
for ind in result["population"]:
	print(ind)

func = [i.objectives for i in result["population"]]
function1 = [i[0].value for i in func]
function2 = [i[1].value for i in func]
plt.xlabel('Function score [MAX]', fontsize=15)
plt.ylabel('Function cost (SP) [MIN]', fontsize=15)
plt.scatter(function1, function2)
plt.show()


