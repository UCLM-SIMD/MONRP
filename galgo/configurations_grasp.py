def generate_configurations():
	# grasp 1 grasp 10 10 best_first_neighbor 5
	# opciones------------------------------------------------------------------
	dataset_problems=[
		1,
		2
	]
	algorithms=[
		"grasp"
	]
	iterations =[
		20,50,100,200,300
	]
	solutions_per_iteration_list = [
		20,40,60,80,100
	]
	local_search_types=[
		"best_first_neighbor"
	]
	seed=10

	f = open("configs.txt", "w")
	returnStr = ''
	type="grasp"
	for dataset_problem in dataset_problems:
		for algorithm in algorithms:
			for iteration in iterations:
				for solutions_per_iteration in solutions_per_iteration_list:
					for local_search_type in local_search_types:
						returnStr = type + " " + str(dataset_problem) + ' ' + str(algorithm)+ ' ' \
									+ str(iteration) + ' ' + str(solutions_per_iteration) + ' ' + str(local_search_type) + \
									' ' + str(seed)  + '\n'
						f.write(returnStr)
	f.close()

generate_configurations()

