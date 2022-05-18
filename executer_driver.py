from algorithms.EDA.FEDA.feda_algorithm import FEDAAlgorithm
from algorithms.EDA.PBIL.pbil_algorithm import PBILAlgorithm
from algorithms.EDA.UMDA.umda_algorithm import UMDAAlgorithm
from algorithms.EDA.bivariate.MIMIC.mimic_algorithm import MIMICAlgorithm
from algorithms.GRASP.GRASP import GRASP
from algorithms.genetic.geneticnds.geneticnds_algorithm import GeneticNDSAlgorithm
from algorithms.genetic.nsga2.nsga2_algorithm import NSGA2Algorithm
from algorithms.genetic.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm
import argparse
import os

<<<<<<< HEAD
OUTPUT_FOLDER = "output/"
=======
from datasets.Dataset import Dataset
>>>>>>> 19c7836f (ahora todos los resultados se almacenan en results.json con un id unico para cada conjunto de parametros de lanzamiento)

curpath = os.path.abspath(os.curdir)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', nargs="+",
                    help='<Required> configuration', required=False)

params = parser.parse_args().config[0].split()  # sh galgo
# params = parser.parse_args().config  # local
print(params)
<<<<<<< HEAD
if (params[0] == "genetic"):
    # "-c genetic geneticnds p1 4 20 100 10000 tournament 2 onepoint 0.8 flip1bit 0.1 elitism 5 D 20"
    # "-c genetic nsgaii p1 4 20 100 10000 tournament 2 onepoint 0.8 flip1bit 0.1 elitism 5 D 20"
    # algorithmtype algorithm dataset seed population_length max_generations max_evaluations selection selection_candidates crossover
    #  crossover_prob mutation mutation_prob replacement num_execs dependencies subset_size
=======
if(params[0] == "genetic"):
    # "-c genetic geneticnds p1 4 20 100 10000 tournament 2 onepoint 0.8 flip1bit 0.1 elitism 5 D"
    # "-c genetic nsgaii p1 4 20 100 10000 tournament 2 onepoint 0.8 flip1bit 0.1 elitism 5 D"
    # algorithmtype algorithm dataset seed population_length max_generations max_evaluations selection selection_candidates crossover
    #  crossover_prob mutation mutation_prob replacement num_execs
>>>>>>> 19c7836f (ahora todos los resultados se almacenan en results.json con un id unico para cada conjunto de parametros de lanzamiento)
    algorithm_name = str(params[1])
    if algorithm_name == "geneticnds":
        algorithm_model = GeneticNDSAlgorithm
    elif algorithm_name == "nsgaii":
        #algorithm_model = NSGAIIAlgorithm
        algorithm_model = NSGA2Algorithm

    dataset_name, seed, pop_length, max_gens, max_evaluations, sel_scheme, selection_candidates, cross_scheme, \
<<<<<<< HEAD
    cross_prob, mut_scheme, mut_prob, repl_scheme, execs, dependencies, subset_size = \
        [str(params[2]), int(params[3]), int(params[4]), int(params[5]), int(params[6]), str(params[7]), int(params[8]),
         str(params[9]), float(params[10]), str(params[11]), float(params[12]), str(params[13]),
         str(params[14]), str(params[15]), int(params[16])]
=======
    cross_prob, mut_scheme, mut_prob, repl_scheme, execs, dependencies = \
        [str(params[2]), int(params[3]), int(params[4]), int(params[5]), int(params[6]), str(params[7]), int(params[8]),
         str(params[9]), float(params[10]), str(params[11]), float(params[12]), str(params[13]), str(params[14]),str(params[15])]
>>>>>>> 19c7836f (ahora todos los resultados se almacenan en results.json con un id unico para cada conjunto de parametros de lanzamiento)

    tackle_dependencies = True if dependencies == 'D' else False
    algorithm = algorithm_model(dataset_name=dataset_name, random_seed=seed, population_length=pop_length,
                                max_generations=max_gens, max_evaluations=max_evaluations,
<<<<<<< HEAD
                                selection=sel_scheme, crossover=cross_scheme, crossover_prob=cross_prob,
                                mutation=mut_scheme,
                                mutation_prob=mut_prob, replacement=repl_scheme, execs=execs,
                                tackle_dependencies=tackle_dependencies, subset_size=subset_size)


elif (params[0] == "grasp"):
    # "-c grasp grasp p1 5 10 10 10000 stochastically best_first_neighbor_random None 5 D 20"
    # algorithmtype algorithm dataset seed iterations solutions_per_iteration max_evaluations init_type
    # local_search_type path_relinking_mode num_execs tackle_dependencies subset_size
    algorithm_model = GRASP

    algorithm_name, dataset_name, seed, iterations, solutions_per_iteration, max_evaluations, init_type, \
    local_search_type, path_relinking, execs, dependencies, subset_size = \
        [str(params[1]), str(params[2]), int(params[3]),
         int(params[4]), int(params[5]), int(params[6]), str(params[7]), str(params[8]),
         str(params[9]), str(params[10]), str(params[11]), int(params[12])]

    path_relinking = "after_local" if path_relinking in ['PR', 'after_local'] else None
    tackle_dependencies = True if dependencies == 'D' else False
    algorithm = algorithm_model(dataset_name=dataset_name, iterations=iterations,
                                solutions_per_iteration=solutions_per_iteration,
                                max_evaluations=max_evaluations, init_type=init_type,
                                local_search_type=local_search_type,
                                path_relinking_mode=path_relinking, seed=seed, execs=execs,
                                tackle_dependencies=tackle_dependencies, subset_size=subset_size)


elif (params[0] == "eda"):
    if (params[1] == "umda"):
        # "-c eda umda p1 5 100 300 10000 2 nds elitism 5 D 20"
        # "-c eda umda p1 5 100 300 10000 2 monoscore elitism 5 d 20"
        # algorithmtype algorithm dataset seed numpop gens max_evaluations selinds
        # selscheme replscheme num_execs D subset_size
        algorithm_model = UMDAAlgorithm

        algorithm_name, dataset_name, seed, numpop, gens, max_evaluations, selinds, selscheme, replscheme, \
        execs, dependencies, subset_size = \
            [str(params[1]), str(params[2]), int(params[3]), int(params[4]), int(params[5]), int(params[6]),
             int(params[7]), str(params[8]), str(params[9]), int(params[10]), str(params[11]), int(params[12])]
        tackle_dependencies = True if dependencies == 'D' else False

        algorithm = algorithm_model(dataset_name=dataset_name, population_length=numpop,
                                    max_generations=gens, max_evaluations=max_evaluations, selected_individuals=selinds,
                                    selection_scheme=selscheme, replacement_scheme=replscheme, random_seed=seed,
                                    execs=execs, tackle_dependencies=tackle_dependencies, subset_size=subset_size)
        # filepath = "output/metrics/umda-"+algorithm.file


    elif (params[1] == "pbil"):
        # "-c eda pbil p1 5 100 300 10000 0.1 0.1 0.1 5 D 20"
        # algorithmtype algorithm dataset seed numpop gens max_evaluations lr mutprob
        # mutshift num_execs dependencies subset_size
        algorithm_model = PBILAlgorithm

        algorithm_name, dataset_name, seed, numpop, gens, max_evaluations, lr, mutprob,\
        mutshift, execs, dependencies, subset_size = \
            [str(params[1]), str(params[2]), int(params[3]),
             int(params[4]), int(params[5]), int(params[6]), float(params[7]),
             float(params[8]), float(params[9]), float(params[10]), str(params[11]),  int(params[12])]
        tackle_dependencies = True if dependencies == 'D' else False
        algorithm = algorithm_model(dataset_name=dataset_name, population_length=numpop,
                                    max_evaluations=max_evaluations,
                                    max_generations=gens, learning_rate=lr,
                                    mutation_prob=mutprob, mutation_shift=mutshift, random_seed=seed, execs=execs,
                                    tackle_dependencies=tackle_dependencies, subset_size=subset_size)
        # filepath = "output/metrics/pbil-"+algorithm.file


    elif (params[1] == 'feda'):
        # -c eda feda p1 5 100 300 10000  5 D
        # algorithmtype algorithm dataset seed numpop gens max_evaluations num_execs dependencies subset_size
        algorithm_model = FEDAAlgorithm

        algorithm_name, dataset_name, seed, numpop, gens, max_evaluations, execs,\
        dependencies, subset_size, sel_scheme = \
            [str(params[1]), str(params[2]), int(params[3]),
             int(params[4]), int(params[5]), int(params[6]),
             int(params[7]), str(params[8]),  int(params[9]), str(params[10])]
        tackle_dependencies = True if dependencies == 'D' else False

        algorithm = algorithm_model(dataset_name=dataset_name, population_length=numpop,
                                    max_evaluations=max_evaluations,
                                    max_generations=gens, selection_scheme=sel_scheme, random_seed=seed, execs=execs,
                                    tackle_dependencies=tackle_dependencies, subset_size=subset_size)

    elif (params[1] == 'mimic'):
        # -c eda mimic p1 5 50 20 0 replacement 50 nds 10 D 10

        algorithm_model = MIMICAlgorithm


        algorithm_name, dataset_name, seed, numpop, gens, max_evaluations, \
        replscheme, selinds, selcheme, execs, dependencies, subset_size = \
            [str(params[1]), str(params[2]), int(params[3]),
             int(params[4]), int(params[5]), int(params[6]),
             str(params[7]), int(params[8]), str(params[9]), int(params[10]), str(params[11]),
             int(params[12])]
        tackle_dependencies = True if dependencies == 'D' else False

        algorithm = algorithm_model(dataset_name=dataset_name, random_seed=seed,
                                    tackle_dependencies=tackle_dependencies, population_length=numpop,
                                    max_generations=gens, max_evaluations = max_evaluations,
                                    selected_individuals=selinds, selection_scheme=selcheme,
                                    replacement_scheme=replscheme, execs=execs, subset_size=subset_size)

algorithm.executer.execute(output_folder=OUTPUT_FOLDER)
=======
                                selection=sel_scheme, crossover=cross_scheme, crossover_prob=cross_prob, mutation=mut_scheme,
                                mutation_prob=mut_prob, replacement=repl_scheme, execs=execs, tackle_dependencies=tackle_dependencies)
    filepath = "output/metrics/results.json"

elif(params[0] == "grasp"):
    # -c grasp grasp p1 5 10 10 10000 stochastically best_first_neighbor_random None 5 D
    # algorithmtype algorithm dataset seed iterations solutions_per_iteration max_evaluations init_type local_search_type path_relinking_mode num_execs
    algorithm_model = GRASP

    algorithm_name, dataset_name, seed, iterations, solutions_per_iteration, max_evaluations, init_type, \
    local_search_type, path_relinking, execs, dependencies = \
        [str(params[1]), str(params[2]), int(params[3]),
            int(params[4]), int(params[5]), int(params[6]), str(params[7]), str(params[8]),
         str(params[9]), str(params[10]), str(params[11])]

    tackle_dependencies = True if dependencies == 'D' else False
    algorithm = algorithm_model(dataset=Dataset(dataset_name), iterations=iterations, solutions_per_iteration=solutions_per_iteration,
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

# try:
algorithm.executer.execute(executions=int(algorithm.num_executions), file_path=filepath)
# except:
#    print("wrong algorithm type")
>>>>>>> 19c7836f (ahora todos los resultados se almacenan en results.json con un id unico para cada conjunto de parametros de lanzamiento)
