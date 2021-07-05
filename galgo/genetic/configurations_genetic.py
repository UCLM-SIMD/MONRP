def generate_configurations():
    # genetic geneticnds dataset1 4 20 100 tournament 2 onepoint 0.8 flip1bit 0.1 elitism
    # opciones------------------------------------------------------------------
    seed = 10
    population_lengths = [
        20,
        30, 40,
        100,
        200,
    ]
    generations = [
                   100,
                   200,
                   300,
                   500,
                   1000,
                   #2000,
                   ]
    dataset_problems = [
        "1",
        "2",
        "s1",
        "s2",
        "s3",
    ]
    algorithms = [
        # "genetic",
        #"geneticnds",
         "nsgaii"
    ]
    selection_schemes = ["tournament",
                         ]
    selection_candidates = [2]
    crossover_schemes = ["onepoint"
                         ]
    crossover_probs = [
        0.6, 
        0.8, 
        0.85, 0.9
                       ]
    mutation_schemes = ["flip1bit",
                        "flipeachbit"
                        ]
    mutation_probs = [
        0, 0.05,
        #1/20,
        1/40,
        #1/80,
        #1/100,
        #1/140,
        0.1, 0.2, 0.5, 0.7, 1
    ]
    replacement_schemes = ["elitism",
                           #"elitismnds"
                           ]

    f = open("configs_genetic.txt", "w")
    returnStr = ''
    type = "genetic"
    for dataset_problem in dataset_problems:
        for selected_algorithm in algorithms:
            for population_length in population_lengths:
                for generation in generations:
                    for selection_scheme in selection_schemes:
                        for selection_candidate in selection_candidates:
                            for crossover_scheme in crossover_schemes:
                                for crossover_prob in crossover_probs:
                                    for mutation_scheme in mutation_schemes:
                                        for mutation_prob in mutation_probs:
                                            for replacement_scheme in replacement_schemes:
                                                returnStr = type + " " + str(selected_algorithm) + ' ' + str(dataset_problem) + ' ' + str(seed) \
                                                    + ' ' + str(population_length) + ' ' + str(generation) + \
                                                    ' ' + str(selection_scheme) + ' ' + str(selection_candidate) + ' ' + str(crossover_scheme) + ' ' \
                                                    + str(crossover_prob) + ' ' + str(mutation_scheme) + ' ' \
                                                    + str(mutation_prob) + ' ' + \
                                                    str(replacement_scheme) + '\n'
                                                if(len(replacement_schemes) > 1 and len(algorithms) > 1):
                                                    if (replacement_scheme == "elitismnds" and selected_algorithm == "geneticnds") or \
                                                            (replacement_scheme != "elitismnds"):
                                                        f.write(returnStr)
                                                else:
                                                    f.write(returnStr)

    f.close()


generate_configurations()
