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
        "mimic"
    ]
    population_length = [
        # 20,
        50,
        100,
        200,  # 500
    ]
    max_generations = [
        # 20,
        50,
        100,
        200,  # 500
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

    f = open("configs_mimic.txt", "w")
    returnStr = ''
    type = "eda"
    for dataset_problem in dataset_problems:
        for algorithm in algorithms:
            for dependencies in tackle_dependencies:
                for pop in population_length:
                    for max_generation in max_generations:
                        for evaluation in evaluations:
                            for selected_individual in selected_individuals:
                                for selection_scheme in selection_schemes:
                                    for replacement_scheme in replacement_schemes:
                                        returnStr = type + ' ' + str(algorithm) + " " + str(dataset_problem) + ' ' + \
                                            str(dependencies) + ' ' + \
                                            str(seed) + ' ' + str(pop) + ' ' + \
                                            str(max_generation) + ' ' + \
                                            str(evaluation) + ' ' + \
                                            str(selected_individual) + ' ' + \
                                            str(selection_scheme) + ' ' + \
                                            str(replacement_scheme) + '\n'
                                        f.write(returnStr)
    f.close()


generate_configurations()
