from genetic.genetic_algorithm import GeneticAlgorithm
from genetic_nds.genetic_nds_algorithm import GeneticNDSAlgorithm
from nsgaii.nsgaii_algorithm import NSGAIIAlgorithm


def executer(algorithm, iterations=10):
	if isinstance(algorithm, GeneticAlgorithm):
		executerGenetic(algorithm, iterations)

	elif isinstance(algorithm, GeneticNDSAlgorithm):
		executerGeneticNDS(algorithm, iterations)

	elif isinstance(algorithm, NSGAIIAlgorithm):
		executerNSGAII(algorithm, iterations)


def executerGenetic(algorithm, iterations):
	avg_time = 0
	avg_best_individual_total_score = 0
	open('output/executer_genetic.txt', 'w').close()
	f = open("output/executer_genetic.txt", "a")
	for i in range(0, iterations):
		print("Executing iteration: ", i + 1)
		result = algorithm.run()
		avg_time += result["time"]
		avg_best_individual_total_score += result["best_individual"].total_score
		f.write("Time(s): " + str(result["time"]) + ", Best Individual Total Score: " + str(
			result["best_individual"].total_score) + "\n")

	f.write("\n")
	f.write("AVERAGE:")
	f.write("Average Time(s): " + str(avg_time / iterations) + ", Average Best Individual Total Score: " + str(
		avg_best_individual_total_score))


def executerGeneticNDS(algorithm, iterations):
	avg_time = 0
	avg_hv = 0
	avg_spread = 0
	avg_best_individual_total_score = 0
	open('output/executer_genetic_nds.txt', 'w').close()
	f = open("output/executer_genetic_nds.txt", "a")
	for i in range(0, iterations):
		print("Executing iteration: ", i + 1)
		result = algorithm.run()
		avg_time += result["time"]
		avg_hv += result["hv"]
		avg_spread += result["spread"]
		avg_best_individual_total_score += result["best_individual"].total_score
		f.write("Time(s): " + str(result["time"]) + ", HV: " + str(result["hv"]) + ", Spread: " + str(
			result["spread"]) + "\n"
				+ ", Best Individual Total Score: " + str(result["best_individual"].total_score) + "\n")

	f.write("\n")
	f.write("AVERAGE:")
	f.write("Average Time(s): " + str(avg_time / iterations) + ", HV: " + str(avg_hv / iterations) + ", Spread: " + str(
		avg_spread / iterations) + ", Average Best Individual Total Score: " + str(
		avg_best_individual_total_score))


def executerNSGAII(algorithm, iterations):
	avg_time = 0
	avg_hv = 0
	avg_spread = 0
	open('output/executer_nsgaii.txt', 'w').close()
	f = open("output/executer_nsgaii.txt", "a")
	for i in range(0, iterations):
		print("Executing iteration: ", i + 1)
		result = algorithm.run()
		avg_time += result["time"]
		avg_hv += result["hv"]
		avg_spread += result["spread"]
		f.write("Time(s): " + str(result["time"]) + ", HV: " + str(result["hv"]) + ", Spread: " + str(
			result["spread"]) + "\n")

	f.write("\n")
	f.write("AVERAGE:")
	f.write("Average Time(s): " + str(avg_time / iterations) + ", HV: " + str(avg_hv / iterations) + ", Spread: " + str(
		avg_spread / iterations))
