from algorithms.genetic.genetic_algorithm import GeneticAlgorithm
from algorithms.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
from dataset1 import generate_dataset1_genes
from algorithms.genetic_nds.genetic_nds_algorithm import GeneticNDSAlgorithm
from dataset2 import generate_dataset2_genes
from models.problem import Problem
import matplotlib.pyplot as plt
import pandas as pd
# generar los genes de los requisitos------------------------------------------------------------------
genes=generate_dataset2_genes()
# score=MAX; cost=MIN;
objectives_minimization=["MAX","MIN"]

# instanciar el problema------------------------------------------------------------------
problem=Problem(genes,objectives_minimization)

# crear seed random------------------------------------------------------------------
seed=54
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
#algorithm=NSGAIIAlgorithm(problem,random_seed=seed,population_length=20,max_generations=100,crossover_prob=0.85,mutation_prob=0.1,replacement="elitism")
algorithm=NSGAIIAlgorithm(problem,random_seed=seed,population_length=30,max_generations=200,crossover_prob=0.6,
						   crossover="onepoint",mutation_prob=0.05,mutation="flipeachbit",replacement="elitism")

result=algorithm.run()
print("Time: ",result["time"])
print("AvgValue: ",result["avgValue"])
if "numSolutions" in result:
	print("NumSolutions: ",result["numSolutions"])
	print("HV: ",result["hv"])
	print("Spread: ",result["spread"])
	print("Spacing: ",result["spacing"])

for ind in result["population"]:
	counter=0
	for other_ind in result["population"]:
		if(ind.dominates(other_ind)):
			counter+=1
			#print(ind)
			#print(other_ind,"--------")
	#print(counter)

'''
for ind in result["population"]:
	isRepeated=False
	print(ind.print_genes())
	same_genes=0
	for other_ind in result["population"]:
		if ind.print_genes() == other_ind.print_genes():
			same_genes+=1
	if same_genes > 1:
		print(same_genes)
	#print(same_genes)

	total_counter = 0
	for other_ind in result["population"]:
		counter = 0
		for i in range(0,len(ind.genes)):
			if(ind.genes[i].included==other_ind.genes[i].included):
				counter+=1
		if counter >=20:
			#print(other_ind)
			total_counter+=1
	#print(total_counter)
'''

	#print(ind)
#print(total_counter)

#print("Front 0:")
#for ind in result["population"]:
	#print(ind)

func = [i.objectives for i in result["population"]]
function1 = [i[0].value for i in func]
function2 = [i[1].value for i in func]

algorithm=GeneticNDSAlgorithm(problem,random_seed=seed,population_length=20,max_generations=100,crossover_prob=0.6,
							 crossover="onepoint", mutation_prob=0.7,mutation="flip1bit",replacement="elitismnds")
result=algorithm.run()
print("Time: ",result["time"])
print("AvgValue: ",result["avgValue"])
if "numSolutions" in result:
	print("NumSolutions: ",result["numSolutions"])
	print("HV: ",result["hv"])
	print("Spread: ",result["spread"])
	print("Spacing: ",result["spacing"])
func = [i.objectives for i in result["population"]]
function3 = [i[0].value for i in func]
function4 = [i[1].value for i in func]


algorithm=GeneticNDSAlgorithm(problem,random_seed=seed,population_length=20,max_generations=200,crossover_prob=0.9,
							  crossover="onepoint", mutation_prob=0.7,mutation="flip1bit",replacement="elitism")
result=algorithm.run()
func = [i.objectives for i in result["population"]]
function5 = [i[0].value for i in func]
function6 = [i[1].value for i in func]

plt.xlabel('Valor', fontsize=12)
plt.ylabel('Coste', fontsize=12)
plt.scatter(function1, function2, marker='o',label="NSGA-II",c="#3cb371")
plt.scatter(function3, function4, marker='x',label="GeneticNDS_elitismNDS",c="#ff0000")
plt.scatter(function5, function6, marker='+',label="GeneticNDS_elitism",c="#4169e1")
plt.legend(loc="upper left")
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('plots/pareto_dataset2.svg', dpi=100)


