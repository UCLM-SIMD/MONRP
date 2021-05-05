from algorithms.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
from algorithms.genetic_nds.genetic_nds_algorithm import GeneticNDSAlgorithm
from datasets.dataset2 import generate_dataset2_genes
from models.problem import Problem
import matplotlib.pyplot as plt

# generar los genes de los requisitos------------------------------------------------------------------
genes=generate_dataset2_genes()
objectives_minimization=["MAX","MIN"]

# instanciar el problema------------------------------------------------------------------
problem=Problem(genes,objectives_minimization)

# crear seed random------------------------------------------------------------------
seed=54
#random.seed(10)

# iniciar------------------------------------------------------------------
print("Running...")
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


