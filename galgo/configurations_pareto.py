def generate_configurations():
    # grasp grasp 1 5 10 10 best_first_neighbor
    # opciones------------------------------------------------------------------
    f = open("configs_pareto.txt", "w")

    algorithms = [
        "grasp grasp 2 5 300 100 uniform best_first_neighbor_sorted_score None",
        "grasp grasp 2 5 300 100 uniform None None",
        #"grasp grasp 1 5 20 20 uniform best_first_neighbor_sorted_score None",
    ]
    for alg in algorithms:
        f.write(alg+ '\n')

    f.close()


generate_configurations()
