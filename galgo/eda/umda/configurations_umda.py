def generate_configurations():
    # eda umda s1 5 100 300 50 nds replacement
    # opciones------------------------------------------------------------------
    dataset_problems = [
        "1",
        "2",
        # "s1",
        # "s2",
        "s3",
    ]
    algorithms = [
        "umda"
    ]
    population_length = [
        # 20,
        50,
        100, 200,  # 500
    ]
    max_generations = [
        # 20,
        50,
        100, 200,  # 500
    ]
    evaluations = [
        # 10000,
        0
    ]
    selected_individuals = [
        20, 50
    ]
    selection_schemes = [
        "nds",
        # "monoscore"
    ]
    replacement_schemes = [
        "replacement",
        "elitism"
    ]
    seed = 10

    f = open("configs_umda.txt", "w")
    returnStr = ''
    type = "eda"
    for dataset_problem in dataset_problems:
        for algorithm in algorithms:
            for pop in population_length:
                for max_generation in max_generations:
                    for evaluation in evaluations:
                        for selected_individual in selected_individuals:
                            for selection_scheme in selection_schemes:
                                for replacement_scheme in replacement_schemes:
                                    returnStr = type + ' ' + str(algorithm) + " " + str(dataset_problem) + ' ' + \
                                        str(seed) + ' ' + str(pop) + ' ' + \
                                        str(max_generation) + ' ' + \
                                        str(evaluation) + ' ' + \
                                        str(selected_individual) + ' ' + \
                                        str(selection_scheme) + ' ' + \
                                        str(replacement_scheme) + '\n'
                                    f.write(returnStr)
    f.close()


generate_configurations()
