def generate_configurations():
	# opciones------------------------------------------------------------------
	seed=10
	population_lengths=[20
		,30,40
		 ]
	generations=[100
		,200,300
		]
	dataset_problems=["dataset1"
		,"dataset2"
					  ]
	algorithms=["genetic"
		,"geneticnds","nsgaii"
		]
	selection_schemes = ["tournament",
						 ]
	crossover_schemes = ["onepoint"
						 ]
	crossover_probs=[0.6,0.8
		,0.85,0.9
		]
	mutation_schemes=["flip1bit",
			   "flipeachbit"
			   ]
	mutation_probs=[0
		,0.05,0.1,0.2,0.5,0.7,1
		 ]
	replacement_schemes=["elitism"
		,"elitismnds"
		]

	f = open("configs.txt", "w")
	returnStr = ''
	for dataset_problem in dataset_problems:
		for selected_algorithm in algorithms:
			for population_length in population_lengths:
				for generation in generations:
					for selection_scheme in selection_schemes:
						for crossover_scheme in crossover_schemes:
							for crossover_prob in crossover_probs:
								for mutation_scheme in mutation_schemes:
									for mutation_prob in mutation_probs:
										for replacement_scheme in replacement_schemes:
											if (replacement_scheme == replacement_schemes[1] and selected_algorithm == algorithms[
												1]) or \
													(replacement_scheme != replacement_schemes[1]):
												returnStr = str(dataset_problem) + ' ' + str(seed) + ' ' + str(
													selected_algorithm) + ' ' + str(population_length) + ' ' + str(generation) +\
															' ' + str(selection_scheme) + ' ' + str(crossover_scheme)+ ' ' \
															+ str(crossover_prob) + ' ' + str(mutation_scheme) + ' '\
															+ str(mutation_prob) + ' ' + str(replacement_scheme) + '\n'
												f.write(returnStr)

	f.close()

generate_configurations()

