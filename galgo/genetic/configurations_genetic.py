def generate_configurations():
    tackle_dependencies = [
        # False,
        True
    ]
    dataset_problems = [
        "p1",
        "p2",
        # "s1",
        # "s2",
        # "s3",
        # "c1",
        # "c2",
        # "c3",
        # "c4",
        # "c5",
        # "c6",
        # "c1",
        # "a1",
        # "a2",
        # "a3",
        # "a4",
    ]
    algorithms = [
        # "genetic",
        "geneticnds",
        "nsgaii"
    ]
    seed = 10
    population_lengths = [
        # 20,
        #30, 40,
        50,
        100,
        # 500
    ]
    generations = [
        100,
        200,
        # 300,
        # 500,
        # 1000,
        # 2000,
    ]
    evaluations = [
        # 10000,
        0
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
        # 0,
        # 0.05,
        # 1/20,
        # 1/40,
        # 1/80,
        # 1/100,
        # 1/140,
        0.1, 0.2, 0.5, 0.7,
        1
    ]
    replacement_schemes = ["elitism",
                           # "elitismnds"
                           ]

    f = open("configs_genetic.txt", "w")
    returnStr = ''
    type = "genetic"
    for dataset_problem in dataset_problems:
        for selected_algorithm in algorithms:
            for dependencies in tackle_dependencies:
                for population_length in population_lengths:
                    for generation in generations:
                        for evaluation in evaluations:
                            for selection_scheme in selection_schemes:
                                for selection_candidate in selection_candidates:
                                    for crossover_scheme in crossover_schemes:
                                        for crossover_prob in crossover_probs:
                                            for mutation_scheme in mutation_schemes:
                                                for mutation_prob in mutation_probs:
                                                    for replacement_scheme in replacement_schemes:
                                                        returnStr = type + " " + str(selected_algorithm) + ' ' + str(dataset_problem) + ' ' + \
                                                            str(dependencies) + ' ' + str(seed) \
                                                            + ' ' + str(population_length) + ' ' + str(generation) + \
                                                            ' ' + str(evaluation) + \
                                                            ' ' + str(selection_scheme) + ' ' + str(selection_candidate) + ' ' + str(crossover_scheme) + ' ' \
                                                            + str(crossover_prob) + ' ' + str(mutation_scheme) + ' ' \
                                                            + str(mutation_prob) + ' ' + \
                                                            str(replacement_scheme) + '\n'
                                                        if(len(replacement_schemes) > 1 and len(algorithms) > 1):
                                                            if (replacement_scheme == "elitismnds" and selected_algorithm == "geneticnds") or \
                                                                    (replacement_scheme != "elitismnds"):
                                                                f.write(
                                                                    returnStr)
                                                        else:
                                                            f.write(returnStr)

    f.close()


generate_configurations()
