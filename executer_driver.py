from algorithms.EDA.FEDA.feda_algorithm import FEDAAlgorithm
from algorithms.EDA.PBIL.pbil_algorithm import PBILAlgorithm
from algorithms.EDA.UMDA.umda_algorithm import UMDAAlgorithm
from algorithms.GRASP.GRASP import GRASP
from algorithms.genetic.geneticnds.geneticnds_algorithm import GeneticNDSAlgorithm
from algorithms.genetic.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
import argparse
import os

from datasets.Dataset import Dataset

curpath = os.path.abspath(os.curdir)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', nargs="+",
                    help='<Required> configuration', required=False)


params = parser.parse_args().config[0].split()  # sh galgo
# params = parser.parse_args().config  # local
print(params)
if(params[0] == "genetic"):
    # "-c genetic geneticnds p1 4 20 100 10000 tournament 2 onepoint 0.8 flip1bit 0.1 elitism 5 D"
    # "-c genetic nsgaii p1 4 20 100 10000 tournament 2 onepoint 0.8 flip1bit 0.1 elitism 5 D"
    # algorithmtype algorithm dataset seed population_length max_generations max_evaluations selection selection_candidates crossover
    #  crossover_prob mutation mutation_prob replacement num_execs
    algorithm_name = str(params[1])
    if algorithm_name == "geneticnds":
        algorithm_model = GeneticNDSAlgorithm
    elif algorithm_name == "nsgaii":
        algorithm_model = NSGAIIAlgorithm

    dataset_name, seed, pop_length, max_gens, max_evaluations, sel_scheme, selection_candidates, cross_scheme, \
    cross_prob, mut_scheme, mut_prob, repl_scheme, execs, dependencies = \
        [str(params[2]), int(params[3]), int(params[4]), int(params[5]), int(params[6]), str(params[7]), int(params[8]),
         str(params[9]), float(params[10]), str(params[11]), float(params[12]), str(params[13]), str(params[14]),str(params[15])]

    tackle_dependencies = True if dependencies == 'D' else False
    algorithm = algorithm_model(dataset_name=dataset_name, random_seed=seed, population_length=pop_length,
                                max_generations=max_gens, max_evaluations=max_evaluations,
                                selection=sel_scheme, crossover=cross_scheme, crossover_prob=cross_prob, mutation=mut_scheme,
                                mutation_prob=mut_prob, replacement=repl_scheme, execs=execs, tackle_dependencies=tackle_dependencies)
    filepath = "output/metrics/results.json"

elif(params[0] == "grasp"):
    # "-c grasp grasp p1 5 10 10 10000 stochastically best_first_neighbor_random None 5 D"
    # algorithmtype algorithm dataset seed iterations solutions_per_iteration max_evaluations init_type local_search_type path_relinking_mode num_execs
    algorithm_model = GRASP

    algorithm_name, dataset_name, seed, iterations, solutions_per_iteration, max_evaluations, init_type, \
    local_search_type, path_relinking, execs, dependencies = \
        [str(params[1]), str(params[2]), int(params[3]),
            int(params[4]), int(params[5]), int(params[6]), str(params[7]), str(params[8]),
         str(params[9]), str(params[10]), str(params[11])]

    tackle_dependencies = True if dependencies == 'D' else False
    algorithm = algorithm_model(dataset_name=dataset_name, iterations=iterations, solutions_per_iteration=solutions_per_iteration,
                                max_evaluations=max_evaluations, init_type=init_type, local_search_type=local_search_type,
                                path_relinking_mode=path_relinking, seed=seed, execs=execs, tackle_dependencies=tackle_dependencies)
    filepath = "output/metrics/results.json"  # +algorithm.file

elif(params[0] == "eda"):
    if(params[1] == "umda"):
        # "-c eda umda p1 5 100 300 10000 2 nds elitism 5 D"
        # "-c eda umda p1 5 100 300 10000 2 monoscore elitism 5 d"
        # algorithmtype algorithm dataset seed numpop gens max_evaluations selinds selscheme replscheme num_execs D
        algorithm_model = UMDAAlgorithm

        algorithm_name, dataset_name, seed,\
                numpop, gens, max_evaluations, selinds, selscheme, replscheme, execs, dependencies= \
            [str(params[1]), str(params[2]), int(params[3]),
                int(params[4]), int(params[5]), int(params[6]), int(params[7]), str(params[8]), str(params[9]),int(params[10]), str(params[11])]
        tackle_dependencies = True if dependencies=='D' else False

        algorithm = algorithm_model(dataset_name=dataset_name, population_length=numpop,
                                    max_generations=gens, max_evaluations=max_evaluations, selected_individuals=selinds,
                                    selection_scheme=selscheme, replacement_scheme=replscheme,
                                    random_seed=seed, execs=execs, tackle_dependencies=tackle_dependencies)
        #filepath = "output/metrics/umda-"+algorithm.file
        filepath = "output/metrics/results.json"

    elif(params[1] == "pbil"):
        # -c eda pbil p1 5 100 300 10000 0.1 0.1 0.1 5 D
        # algorithmtype algorithm dataset seed numpop gens max_evaluations lr mutprob mutshift num_execs dependencies
        algorithm_model = PBILAlgorithm

        algorithm_name, dataset_name, seed, numpop, gens, max_evaluations, lr, mutprob, mutshift, execs, dependencies = \
            [str(params[1]), str(params[2]), int(params[3]),
                int(params[4]), int(params[5]), int(params[6]), float(params[7]),
             float(params[8]), float(params[9]), float(params[10]), str(params[11])]
        tackle_dependencies = True if dependencies == 'D' else False
        algorithm = algorithm_model(dataset_name=dataset_name, population_length=numpop, max_evaluations=max_evaluations,
                                    max_generations=gens, learning_rate=lr,
        mutation_prob=mutprob, mutation_shift=mutshift, random_seed=seed, execs=execs, tackle_dependencies=tackle_dependencies)
        #filepath = "output/metrics/pbil-"+algorithm.file
        filepath = "output/metrics/results.json"

    elif(params[1] == 'feda'):
        # -c eda feda p1 5 100 300 10000  5 D
        # algorithmtype algorithm dataset seed numpop gens max_evaluations num_execs dependencies
        algorithm_model = FEDAAlgorithm

        algorithm_name, dataset_name, seed, numpop, gens, max_evaluations, execs, dependencies = \
            [str(params[1]), str(params[2]), int(params[3]),
             int(params[4]),  int(params[5]), int(params[6]),
             int(params[7]), str(params[8])]
        tackle_dependencies = True if dependencies == 'D' else False

        algorithm = algorithm_model(dataset_name=dataset_name, population_length=numpop, max_evaluations=max_evaluations,
                        max_generations=gens, random_seed=seed, execs=execs,
                        tackle_dependencies=tackle_dependencies)
        # filepath = "output/metrics/pbil-"+algorithm.file
        filepath = "output/metrics/results.json"

# try:
algorithm.executer.execute(executions=int(algorithm.num_executions), file_path=filepath)
# except:
#    print("wrong algorithm type")
