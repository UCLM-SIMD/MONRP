def generate_configurations():
    # eda umda s1 5 100 300 50
    # opciones------------------------------------------------------------------
    dataset_problems = [
        "1",
        "2",
        "s1",
        "s2",
        "s3",
    ]
    algorithms = [
        "umda"
    ]
    population_length = [
        20, 50, 100, 200, 300, 500
    ]
    max_generations = [
        20, 40, 60, 80, 100, 200, 500
    ]
    selected_individuals = [
        20, 40, 60, 80, 100, 200, 500
    ]
    seed = 10

    f = open("configs_umda.txt", "w")
    returnStr = ''
    type = "eda"
    for dataset_problem in dataset_problems:
        for algorithm in algorithms:
            for pop in population_length:
                for max_generation in max_generations:
                    for selected_individual in selected_individuals:
                        returnStr = type + ' ' + str(algorithm) + " " + str(dataset_problem) + ' ' + \
                            str(seed) + ' ' + str(pop) + ' ' + \
                            str(max_generation) + ' ' + \
                            str(selected_individual) + '\n'
                        f.write(returnStr)
    f.close()


generate_configurations()
