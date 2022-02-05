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
        "grasp"
    ]
    iterations = [
        # 20,
        50,
        100,
        200,
        # 300,
        # 500
    ]
    solutions_per_iteration_list = [
        # 20,
        #  40, 60, 80,
        50,
        100,
        200,
        # 500
    ]
    evaluations = [
        # 10000,
        0,
    ]
    init_types = ["stochastically"
                  "uniform"
                  ]
    local_search_types = [
        "None",
        "best_first_neighbor_random",
        # "best_first_neighbor_sorted_score",
        # "best_first_neighbor_sorted_score_r",
        # "best_first_neighbor_sorted_domination",
        "best_first_neighbor_random_domination",
    ]
    path_relinking_types = [
        "None",
        "after_local"
    ]
    seed = 10

    f = open("configs_grasp.txt", "w")
    returnStr = ''
    type = "grasp"
    for dataset_problem in dataset_problems:
        for algorithm in algorithms:
            for dependencies in tackle_dependencies:
                for iteration in iterations:
                    for solutions_per_iteration in solutions_per_iteration_list:
                        for evaluation in evaluations:
                            for init_type in init_types:
                                for local_search_type in local_search_types:
                                    for path_relinking_type in path_relinking_types:
                                        returnStr = type + ' ' + str(algorithm) + " " + str(dataset_problem) + ' ' + \
                                            str(dependencies) + ' ' + \
                                            str(seed) + ' ' + str(iteration) + ' ' + \
                                            str(solutions_per_iteration) + ' ' + \
                                            str(evaluation) + ' ' + \
                                            str(init_type) + ' ' + \
                                            str(local_search_type) + ' ' + \
                                            str(path_relinking_type) + '\n'
                                        f.write(returnStr)
    f.close()


generate_configurations()
