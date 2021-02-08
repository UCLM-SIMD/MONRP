'''
def executer(algorithm, iterations=10):

	if isinstance(algorithm, GeneticAlgorithm):
		print("Running Genetic...")
		executerGenetic(algorithm, iterations)

	elif isinstance(algorithm, GeneticNDSAlgorithm):
		print("Running GeneticNDS...")
		executerGeneticNDS(algorithm, iterations)

	elif isinstance(algorithm, NSGAIIAlgorithm):
		print("Running NSGA-II...")
		executerNSGAII(algorithm, iterations)
'''

def initialize_file():
	print("Running...")
	f = open("output/executer.txt", "w")
	f.write("Algorithm, Population Length, Generations, "
			"Selection Scheme, Selection Candidates, Crossover Scheme, Crossover Probability, Mutation Scheme,"
			" Mutation Probability, Replacement Scheme, Time(s), HV, Spread\n")
	print("File reseted")

'''
def executerGenetic(algorithm, iterations):
	avg_time = 0
	avg_best_individual_total_score = 0
	f = open("output/executer_genetic.txt", "a")

	f.write("Time(s), Best Individual Total Score\n")
	for i in range(0, iterations):
		print("Executing iteration: ", i + 1)
		result = algorithm.run()
		avg_time += result["time"]
		avg_best_individual_total_score += result["best_individual"].total_score
		f.write(str(result["time"]) + ", " + str(
			result["best_individual"].total_score) + "\n")

	# f.write("\n")
	# f.write("AVERAGE:")
	# f.write
	print("Average Time(s): " + str(avg_time / iterations) + ", Average Best Individual Total Score: " + str(
		avg_best_individual_total_score))


def executerGeneticNDS(algorithm, iterations):
	avg_time = 0
	avg_hv = 0
	avg_spread = 0
	avg_best_individual_total_score = 0
	f = open("output/executer_genetic_nds.txt", "a")

	f.write("Time(s), HV, Spread, Best Individual Total Score\n")
	for i in range(0, iterations):
		print("Executing iteration: ", i + 1)
		result = algorithm.run()
		avg_time += result["time"]
		avg_hv += result["hv"]
		avg_spread += result["spread"]
		avg_best_individual_total_score += result["best_individual"].total_score
		f.write(str(result["time"]) + ", " + str(result["hv"]) + ", " + str(
			result["spread"]) + ", " + str(result["best_individual"].total_score) + "\n")

	# f.write("\n")
	# f.write("AVERAGE:")
	# f.write
	print("Average Time(s): " + str(avg_time / iterations) + ", HV: " + str(avg_hv / iterations) + ", Spread: " + str(
		avg_spread / iterations) + ", Average Best Individual Total Score: " + str(
		avg_best_individual_total_score))


def executerNSGAII(algorithm, iterations):
	avg_time = 0
	avg_hv = 0
	avg_spread = 0
	f = open("output/executer_nsgaii.txt", "a")
	f.write("Time(s), HV, Spread\n")

	for i in range(0, iterations):
		print("Executing iteration: ", i + 1)
		result = algorithm.run()
		avg_time += result["time"]
		avg_hv += result["hv"]
		avg_spread += result["spread"]
		f.write(str(result["time"]) + ", " + str(result["hv"]) + ", " + str(
			result["spread"]) + "\n")

	# f.write("\n")
	# f.write("AVERAGE:")
	# f.write
	print("Average Time(s): " + str(avg_time / iterations) + ", HV: " + str(avg_hv / iterations) + ", Spread: " + str(
		avg_spread / iterations))

'''

def executer(algorithm, iterations):
	algorithm_name = algorithm.__class__.__name__
	population_length = algorithm.population_length
	generations = algorithm.max_generations
	selection = algorithm.selection_scheme
	selection_candidates = algorithm.selection_candidates
	crossover = algorithm.crossover_scheme
	crossover_prob = algorithm.crossover_prob
	mutation = algorithm.mutation_scheme
	mutation_prob = algorithm.mutation_prob
	replacement = algorithm.replacement_scheme

	for i in range(0, iterations):
		print("Executing iteration: ", i + 1)
		result = algorithm.run()

		time = str(result["time"]) if "time" in result else '-'
		hv = str(result["hv"]) if "hv" in result else '-'
		spread = str(result["spread"]) if "spread" in result else '-'

		f = open("output/executer.txt", "a")
		f.write(str(algorithm_name) + ", " +
				str(population_length) + ", " +
				str(generations) + ", " +
				str(selection) + ", " +
				str(selection_candidates) + ", " +
				str(crossover) + ", " +
				str(crossover_prob) + ", " +
				str(mutation) + ", " +
				str(mutation_prob) + ", " +
				str(replacement) + ", " +
				str(time) + ", " +
				str(hv) + ", " +
				str(spread) +
				"\n")
		f.close()

	print("End")
