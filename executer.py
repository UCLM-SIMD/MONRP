def initialize_file(file_path):
	#print("Running...")
	f = open(file_path, "w")
	f.write("Dataset,Algorithm,Population Length,Generations,"
			"Selection Scheme,Selection Candidates,Crossover Scheme,Crossover Probability,Mutation Scheme,"
			"Mutation Probability,Replacement Scheme,Time(s),AvgValue,BestAvgValue,BestGeneration,HV,Spread,NumSolutions,Spacing\n")
	#print("File reseted")
	f.close()

def executer(algorithm, dataset, iterations, file_path):
	algorithm_name = algorithm.__class__.__name__
	dataset = dataset
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
		#print("Executing iteration: ", i + 1)
		result = algorithm.run()

		time = str(result["time"]) if "time" in result else 'NaN'
		avgValue = str(result["avgValue"]) if "avgValue" in result else 'NaN'
		bestAvgValue = str(result["bestAvgValue"]) if "bestAvgValue" in result else 'NaN'
		bestGeneration = str(result["best_generation_num"]) if "best_generation_num" in result else 'NaN'
		hv = str(result["hv"]) if "hv" in result else 'NaN'
		spread = str(result["spread"]) if "spread" in result else 'NaN'
		numSolutions = str(result["numSolutions"]) if "numSolutions" in result else 'NaN'
		spacing = str(result["spacing"]) if "spacing" in result else 'NaN'

		f = open(file_path, "a")
		f.write(
				str(dataset) + "," +
				str(algorithm_name) + "," +
				str(population_length) + "," +
				str(generations) + "," +
				str(selection) + "," +
				str(selection_candidates) + "," +
				str(crossover) + "," +
				str(crossover_prob) + "," +
				str(mutation) + "," +
				str(mutation_prob) + "," +
				str(replacement) + "," +
				str(time) + "," +
				str(avgValue) + "," +
				str(bestAvgValue) + "," +
				str(bestGeneration) + "," +
				str(hv) + "," +
				str(spread) +"," +
				str(numSolutions) + "," +
				str(spacing)+
				"\n")
		f.close()

	#print("End")
