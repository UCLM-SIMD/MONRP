from algorithms.EDA.FEDA.feda_algorithm import FEDAAlgorithm
from algorithms.EDA.PBIL.pbil_algorithm import PBILAlgorithm
from algorithms.EDA.UMDA.umda_algorithm import UMDAAlgorithm
from algorithms.EDA.bivariate.MIMIC.mimic_algorithm import MIMICAlgorithm
from algorithms.GRASP.GRASP import GRASP
from algorithms.genetic.geneticnds.geneticnds_algorithm import GeneticNDSAlgorithm
from algorithms.genetic.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
import argparse
import os
curpath = os.path.abspath(os.curdir)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', nargs="+",
                    help='<Required> configuration', required=True)
parser.add_argument('-m', '--mode',
                    help='<Required> mode', required=True)

params = parser.parse_args().config[0].split()  # sh galgo
# params = parser.parse_args().config # debug local
mode = parser.parse_args().mode.split()[0]  # sh galgo
# params = parser.parse_args().config # debug local
print(params, mode)

if mode != "metrics" and mode != "paretos":
    raise Exception("Error, mode not found")


def configure_algorithm(params, mode):

    if(params[0] == "genetic"):
        # -c genetic geneticnds s1 True 4 20 100 10000 tournament 2 onepoint 0.8 flip1bit 0.1 elitism
        # algorithmtype algorithm dataset tackle_dependencies seed population_length max_generations max_evaluations selection selection_candidates crossover
        #  crossover_prob mutation mutation_prob replacement
        algorithm_name = str(params[1])
        if algorithm_name == "geneticnds":
            algorithm_model = GeneticNDSAlgorithm
        elif algorithm_name == "nsgaii":
            algorithm_model = NSGAIIAlgorithm

        dataset_name, tackle_dependencies, seed, pop_length, max_gens, evaluations, sel_scheme, selection_candidates, cross_scheme, cross_prob, mut_scheme, mut_prob, repl_scheme = \
            [str(params[2]), (params[3] == "True"), int(params[4]), int(params[5]), int(params[6]), int(params[7]), str(params[8]), int(params[9]),
             str(params[10]), float(params[11]), str(params[12]), float(params[13]), str(params[14])]

        algorithm = algorithm_model(dataset_name=dataset_name, tackle_dependencies=tackle_dependencies, random_seed=seed, population_length=pop_length,
                                    max_generations=max_gens, max_evaluations=evaluations,
                                    selection=sel_scheme, crossover=cross_scheme, crossover_prob=cross_prob, mutation=mut_scheme,
                                    mutation_prob=mut_prob, replacement=repl_scheme)
        filepath = f"output/{mode}/genetic-"+algorithm.get_file()

    elif(params[0] == "grasp"):
        # -c grasp grasp s1 True 5 10 10 10000 stochastically best_first_neighbor None
        # algorithmtype algorithm dataset tackle_dependencies seed iterations solutions_per_iteration evaluations init_type local_search_type path_relinking_mode
        algorithm_model = GRASP

        algorithm_name, dataset_name, tackle_dependencies, seed, iterations, solutions_per_iteration, evaluations, init_type, local_search_type, path_relinking = \
            [str(params[1]), str(params[2]), (params[3] == "True"), int(params[4]),
                int(params[5]), int(params[6]), int(params[7]), str(params[8]), str(params[9]), str(params[10])]

        algorithm = algorithm_model(dataset=dataset_name, tackle_dependencies=tackle_dependencies, iterations=iterations, solutions_per_iteration=solutions_per_iteration,
                                    max_evaluations=evaluations,
                                    init_type=init_type, local_search_type=local_search_type, path_relinking_mode=path_relinking, seed=seed)
        filepath = f"output/{mode}/grasp-"+algorithm.get_file()

    elif(params[0] == "eda"):
        if(params[1] == "umda"):
            # -c eda umda s1 True 5 100 300 10000 50 nds replacement
            # algorithmtype algorithm dataset tackle_dependencies seed numpop gens max_evaluations selinds selscheme replscheme
            algorithm_model = UMDAAlgorithm

            algorithm_name, dataset_name, tackle_dependencies, seed, numpop, gens, max_evaluations, selinds, selscheme, replscheme = \
                [str(params[1]), str(params[2]), (params[3] == "True"), int(params[4]),
                    int(params[5]), int(params[6]), int(params[7]), int(params[8]), str(params[9]), str(params[10])]

            algorithm = algorithm_model(dataset_name=dataset_name, tackle_dependencies=tackle_dependencies, population_length=numpop,
                                        max_generations=gens, max_evaluations=max_evaluations, selected_individuals=selinds,
                                        selection_scheme=selscheme, replacement_scheme=replscheme, random_seed=seed)
            filepath = f"output/{mode}/umda-"+algorithm.get_file()

        elif(params[1] == "pbil"):
            # -c eda pbil s1 True 5 100 300 10000 0.1 0.1 0.1
            # algorithmtype algorithm dataset tackle_dependencies seed numpop gens evaluations lr mutprob mutshift
            algorithm_model = PBILAlgorithm

            algorithm_name, dataset_name, tackle_dependencies, seed, numpop, gens, evaluations, lr, mutprob, mutshift = \
                [str(params[1]), str(params[2]), (params[3] == "True"), int(params[4]),
                    int(params[5]), int(params[6]), int(params[7]), float(params[8]), float(params[9]), float(params[10])]

            algorithm = algorithm_model(dataset_name=dataset_name, tackle_dependencies=tackle_dependencies, population_length=numpop, max_evaluations=evaluations,
                                        max_generations=gens, learning_rate=lr, mutation_prob=mutprob, mutation_shift=mutshift, random_seed=seed)
            filepath = f"output/{mode}/pbil-"+algorithm.get_file()
        
        elif(params[1] == "mimic"):
            # -c eda mimic s1 True 5 100 300 10000 50 nds replacement
            # algorithmtype algorithm dataset tackle_dependencies seed numpop gens max_evaluations selinds selscheme replscheme
            algorithm_model =MIMICAlgorithm

            algorithm_name, dataset_name, tackle_dependencies, seed, numpop, gens, max_evaluations, selinds, selscheme, replscheme = \
                [str(params[1]), str(params[2]), (params[3] == "True"), int(params[4]),
                    int(params[5]), int(params[6]), int(params[7]), int(params[8]), str(params[9]), str(params[10])]

            algorithm = algorithm_model(dataset_name=dataset_name, tackle_dependencies=tackle_dependencies, population_length=numpop,
                                        max_generations=gens, max_evaluations=max_evaluations, selected_individuals=selinds,
                                        selection_scheme=selscheme, replacement_scheme=replscheme, random_seed=seed)
            filepath = f"output/{mode}/mimic-"+algorithm.get_file()

        elif(params[1] == "feda"):
            # -c eda feda s1 True 5 100 100 10000 nds
            # algorithmtype algorithm dataset tackle_dependencies seed numpop gens evaluations selscheme
            algorithm_model = FEDAAlgorithm

            algorithm_name, dataset_name, tackle_dependencies, seed, numpop, gens, evaluations, selscheme = \
                [str(params[1]), str(params[2]), (params[3] == "True"), int(params[4]),
                    int(params[5]), int(params[6]), int(params[7]), str(params[8])]

            algorithm = algorithm_model(dataset_name=dataset_name, tackle_dependencies=tackle_dependencies, population_length=numpop, max_evaluations=evaluations,
                                        max_generations=gens, selection_scheme=selscheme, random_seed=seed)
            filepath = f"output/{mode}/feda-"+algorithm.get_file()

    return algorithm, filepath


algorithm, filepath = configure_algorithm(params, mode)

if mode == "metrics":
    algorithm.executer.execute(executions=10, file_path=filepath)
elif mode == "paretos":
    algorithm.executer.execute_pareto(file_path=filepath)
