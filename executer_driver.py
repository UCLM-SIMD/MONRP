from algorithms.GRASP.GRASP import GRASP
from algorithms.genetic.genetic_algorithm import GeneticAlgorithm
from algorithms.geneticnds.geneticnds_algorithm import GeneticNDSAlgorithm
from algorithms.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
import argparse
import os
curpath = os.path.abspath(os.curdir)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', nargs="+",
                    help='<Required> configuration', required=False)


params = parser.parse_args().config[0].split()  # sh galgo

if(params[0] == "genetic"):
    # -c genetic geneticnds 1 4 20 100 tournament 2 onepoint 0.8 flip1bit 0.1 elitism
    # algorithmtype algorithm dataset seed population_length max_generations selection selection_candidates crossover
    #  crossover_prob mutation mutation_prob replacement
    algorithm_name = str(params[1])
    if algorithm_name == "genetic":
        algorithm_model = GeneticAlgorithm
    elif algorithm_name == "geneticnds":
        algorithm_model = GeneticNDSAlgorithm
    elif algorithm_name == "nsgaii":
        algorithm_model = NSGAIIAlgorithm

    dataset_name, seed, pop_length, max_gens, sel_scheme, selection_candidates, cross_scheme, cross_prob, mut_scheme, mut_prob, repl_scheme = \
        [str(params[2]), int(params[3]), int(params[4]), int(params[5]), str(params[6]), int(params[7]),
         str(params[8]), float(params[9]), str(params[10]), float(params[11]), str(params[12])]

    algorithm = algorithm_model(dataset_name=dataset_name, random_seed=seed, population_length=pop_length, max_generations=max_gens,
                                selection=sel_scheme, crossover=cross_scheme, crossover_prob=cross_prob, mutation=mut_scheme,
                                mutation_prob=mut_prob, replacement=repl_scheme)
    filepath = "output/genetic-"+algorithm.file

elif(params[0] == "grasp"):
    # -c grasp grasp 1 5 10 10 best_first_neighbor
    # algorithmtype algorithm dataset seed iterations solutions_per_iteration local_search_type
    algorithm_model = GRASP

    algorithm_name, dataset_name, seed, iterations, solutions_per_iteration, local_search_type = \
        [str(params[1]), str(params[2]), int(params[3]),
            int(params[4]), int(params[5]), str(params[6])]

    algorithm = algorithm_model(dataset=dataset_name, iterations=iterations, solutions_per_iteration=solutions_per_iteration,
                                local_search_type=local_search_type, seed=seed)
    filepath = "output/grasp-"+algorithm.file

# try:
algorithm.executer.execute(executions=10, file_path=filepath)
# except:
#    print("wrong algorithm type")