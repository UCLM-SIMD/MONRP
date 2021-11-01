from algorithms.EDA.PBIL.pbil_algorithm import PBILAlgorithm
from algorithms.EDA.UMDA.umda_algorithm import UMDAAlgorithm
from algorithms.GRASP.GRASP import GRASP
from algorithms.genetic.geneticnds.geneticnds_algorithm import GeneticNDSAlgorithm
from algorithms.genetic.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
import argparse
import os
curpath = os.path.abspath(os.curdir)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', nargs="+",
                    help='<Required> configuration', required=False)


# params = parser.parse_args().config[0].split()  # sh galgo
params = parser.parse_args().config  # local
print(params)
if(params[0] == "genetic"):
    # -c genetic geneticnds 1 4 20 100 10000 tournament 2 onepoint 0.8 flip1bit 0.1 elitism
    # algorithmtype algorithm dataset seed population_length max_generations max_evaluations selection selection_candidates crossover
    #  crossover_prob mutation mutation_prob replacement
    algorithm_name = str(params[1])
    if algorithm_name == "geneticnds":
        algorithm_model = GeneticNDSAlgorithm
    elif algorithm_name == "nsgaii":
        algorithm_model = NSGAIIAlgorithm

    dataset_name, seed, pop_length, max_gens, max_evaluations, sel_scheme, selection_candidates, cross_scheme, cross_prob, mut_scheme, mut_prob, repl_scheme = \
        [str(params[2]), int(params[3]), int(params[4]), int(params[5]), int(params[6]), str(params[7]), int(params[8]),
         str(params[9]), float(params[10]), str(params[11]), float(params[12]), str(params[13])]

    algorithm = algorithm_model(dataset_name=dataset_name, random_seed=seed, population_length=pop_length,
                                max_generations=max_gens, max_evaluations=max_evaluations,
                                selection=sel_scheme, crossover=cross_scheme, crossover_prob=cross_prob, mutation=mut_scheme,
                                mutation_prob=mut_prob, replacement=repl_scheme)
    filepath = "output/metrics/genetic-"+algorithm.file

elif(params[0] == "grasp"):
    # -c grasp grasp 1 5 10 10 10000 stochastically best_first_neighbor None
    # algorithmtype algorithm dataset seed iterations solutions_per_iteration max_evaluations init_type local_search_type path_relinking_mode
    algorithm_model = GRASP

    algorithm_name, dataset_name, seed, iterations, solutions_per_iteration, max_evaluations, init_type, local_search_type, path_relinking = \
        [str(params[1]), str(params[2]), int(params[3]),
            int(params[4]), int(params[5]), int(params[6]), str(params[7]), str(params[8]), str(params[9])]

    algorithm = algorithm_model(dataset=dataset_name, iterations=iterations, solutions_per_iteration=solutions_per_iteration,
                                max_evaluations=max_evaluations,
                                init_type=init_type, local_search_type=local_search_type, path_relinking_mode=path_relinking, seed=seed)
    filepath = "output/metrics/grasp-"+algorithm.file

elif(params[0] == "eda"):
    if(params[1] == "umda"):
        # -c eda umda s1 5 100 300 10000 50
        # algorithmtype algorithm dataset seed numpop gens max_evaluations selinds
        algorithm_model = UMDAAlgorithm

        algorithm_name, dataset_name, seed, numpop, gens, max_evaluations, selinds = \
            [str(params[1]), str(params[2]), int(params[3]),
                int(params[4]), int(params[5]), int(params[6]), int(params[7])]

        algorithm = algorithm_model(dataset_name=dataset_name, population_length=numpop,
                                    max_generations=gens, max_evaluations=max_evaluations, selected_individuals=selinds, random_seed=seed)
        filepath = "output/metrics/umda-"+algorithm.file

    elif(params[1] == "pbil"):
        # -c eda pbil s1 5 100 300 10000 0.1 0.1 0.1
        # algorithmtype algorithm dataset seed numpop gens max_evaluations lr mutprob mutshift
        algorithm_model = PBILAlgorithm

        algorithm_name, dataset_name, seed, numpop, gens, max_evaluations, lr, mutprob, mutshift = \
            [str(params[1]), str(params[2]), int(params[3]),
                int(params[4]), int(params[5]), int(params[6]), float(params[7]), float(params[8]), float(params[9])]

        algorithm = algorithm_model(dataset_name=dataset_name, population_length=numpop, max_evaluations=max_evaluations,
                                    max_generations=gens, learning_rate=lr, mutation_prob=mutprob, mutation_shift=mutshift, random_seed=seed)
        filepath = "output/metrics/pbil-"+algorithm.file

# try:
algorithm.executer.execute(executions=10, file_path=filepath)
# except:
#    print("wrong algorithm type")
