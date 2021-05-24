class Executer():
	def __init__(self,algorithm_type="genetic"):
		self.algorithm_type = algorithm_type

	def initialize_file(self,file_path):
		#print("Running...")
		f = open(file_path, "w")
		if self.algorithm_type == "genetic":
			f.write("Dataset,Algorithm,Population Length,Generations,"
					"Selection Scheme,Selection Candidates,Crossover Scheme,Crossover Probability,Mutation Scheme,"
					"Mutation Probability,Replacement Scheme,Time(s),AvgValue,BestAvgValue,BestGeneration,HV,Spread,NumSolutions,Spacing,NumGenerations\n")
			#print("File reseted")
		elif self.algorithm_type == "grasp":
			f.write("Dataset,Algorithm,Iterations,Solutions per Iteration,"
					"Local Search Type,Time(s),AvgValue,BestAvgValue,HV,Spread,NumSolutions,Spacing,NumGenerations\n")
		f.close()

	def reset_file(self,file_path):
		file = open(file_path, "w")
		file.close()

	def execute(self,algorithm, dataset, executions, file_path):
		algorithm_name = algorithm.__class__.__name__
		dataset = dataset
		if self.algorithm_type == "genetic":
			population_length = algorithm.population_length
			generations = algorithm.max_generations
			selection = algorithm.selection_scheme
			selection_candidates = algorithm.selection_candidates
			crossover = algorithm.crossover_scheme
			crossover_prob = algorithm.crossover_prob
			mutation = algorithm.mutation_scheme
			mutation_prob = algorithm.mutation_prob
			replacement = algorithm.replacement_scheme

		elif self.algorithm_type == "grasp":
			iterations = algorithm.iterations
			solutions_per_iteration = algorithm.solutions_per_iteration
			local_search_type = algorithm.local_search_type

		for i in range(0, executions):
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
			num_generations = str(result["num_generations"]) if "num_generations" in result else 'NaN'

			f = open(file_path, "a")
			if self.algorithm_type == "genetic":
				data = str(dataset) + "," + \
					   str(algorithm_name) + "," + \
					   str(population_length) + "," + \
					   str(generations) + "," + \
					   str(selection) + "," + \
					   str(selection_candidates) + "," + \
					   str(crossover) + "," + \
					   str(crossover_prob) + "," + \
					   str(mutation) + "," + \
					   str(mutation_prob) + "," + \
					   str(replacement) + "," + \
					   str(time) + "," + \
					   str(avgValue) + "," + \
					   str(bestAvgValue) + "," + \
					   str(bestGeneration) + "," + \
					   str(hv) + "," + \
					   str(spread) +"," + \
					   str(numSolutions) + "," + \
					   str(spacing)+ "," + \
					   str(num_generations)+ \
					   "\n"

			elif self.algorithm_type == "grasp":
				data = str(dataset) + "," + \
					   str(algorithm_name) + "," + \
					   str(iterations) + "," + \
					   str(solutions_per_iteration) + "," + \
					   str(local_search_type) + ","  + \
					   str(time) + "," + \
						str(avgValue) + "," + \
						str(bestAvgValue) + "," + \
						str(hv) + "," + \
						str(spread) +"," + \
						str(numSolutions) + "," + \
						str(spacing)+ "," + \
						str(num_generations)+ \
						"\n"

			f.write(data)
			f.close()

		#print("End")
