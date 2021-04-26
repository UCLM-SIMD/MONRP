def generate_configurations():
	# opciones------------------------------------------------------------------
	seed=10
	cross=[0.8,0.85,0.9]
	mut=[0,0.05,0.1]
	pop=[20,30,40]
	generations=[100,200,300]
	dataset_problems=["dataset1","dataset2"]
	algorithms=["genetic","geneticnds","nsgaii"]
	replacements=["elitism","elitismnds"]

	#python "$FILE" ${datasets[0]} seed ${algorithms[0]} ${popsize[0]} ${generations[0]} ${crossovers[0]} ${mutations[0]} ${replacements[0]} FILE_PATH

	returnStr = ''
	for c in cross:
		for m in mut:
			for p in pop:
				for g in generations:
					for dataset_problem in dataset_problems:
						for selected_algorithm in algorithms:
							for r in replacements:
								returnStr += str(dataset_problem) + ' '+str(seed) + ' '+str(selected_algorithm) + ' '+ \
											 str(p) + ' ' +str(g) + ' '+str(c) + ' '+ \
											 str(m) + ' ' +str(r)+ '\n'

	print(returnStr)
	return returnStr

generate_configurations()