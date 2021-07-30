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
        "pbil"
    ]
    population_length = [
        #20, 50, 
        100, 200, 300, 500
    ]
    max_generations = [
        20, 40, 60, 80, 100, 200, 500
    ]
    learning_rate = [
        #0.01, 0.05, 
        0.1, 0.2,# 0.5, 1
    ]
    mutation_prob = [
        0, 0.1, 0.2,# 0.5, 1
    ]
    mutation_shift = [
        #0.01, 0.05, 
        0.1, 0.2,# 0.5, 1
    ]
    seed = 10

    f = open("configs_pbil.txt", "w")
    returnStr = ''
    type = "eda"
    for dataset_problem in dataset_problems:
        for algorithm in algorithms:
            for pop in population_length:
                for max_generation in max_generations:
                    for lea in learning_rate:
                        for muprob in mutation_prob:
                            for mushift in mutation_shift:
                                returnStr = type + ' ' + str(algorithm) + " " + str(dataset_problem) + ' ' + \
                                    str(seed) + ' ' + str(pop) + ' ' + \
                                    str(max_generation) + ' ' + \
                                    str(lea) + ' ' + \
                                    str(muprob) + ' ' + \
                                    str(mushift) + '\n'
                                f.write(returnStr)
    f.close()


generate_configurations()
