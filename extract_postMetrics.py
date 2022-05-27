from pymoo.factory import get_performance_indicator
from pymoo.visualization.scatter import Scatter
import json

""" Please fill the experiments hyper-parameters, which will be used to define the which results
will be taken into account to find the reference Pareto for GD+ and UNFR"""

# COMMON HYPER-PARAMETERS #
algorithms = ['GRASP', 'geneticNDS', 'umda', 'pbil', 'feda']  # {'GRASP', 'feda', 'geneticNDS', 'pbil', 'umda'}
dataset = ['p1']  # {'p1','p2','s1','s2','s3','a1','a2','a3','a4','c1','c2','c3','c4','c5','c6'}
max_evaluations = [10000]
population_size = [10]  # called 'solutions_per_iteration' in grasp, and 'population_length' in edas and geneticnds
num_iterations = [10]  # called 'grasp_iterations' in grasp and 'max_generations' in edas and geneticnds
dependencies = ['D']  # {'D','d'} D to tackle deps., 'd' otherwise
seed = 5
num_executions = 5

# geneticNDS hyper-parameters #
selection_candidates = [2]
crossover_prob = [0.8]
mutation_prob = [0.1]
mutation = ['flip1bit']  # {'flip1bit', 'flipeachbit'}
replacement = ['elitism']  # {'elitism', 'elitismnds'}
selection = ['tournament']  # only 'tournament' available
crossover = ['onepoint']  # only 'onepoint' available

# GRASP hyper-parameters #
init_type = ['stochastically']  # {'stochastically', 'uniform'}
path_relinking_mode = ['None', 'PR']  # {'None', 'PR'}
local_search_type = ['best_first_neighbor_random']
# local_search_type values: {'None', 'best_first_neighbor_random','best_first_neighbor_sorted_score',
# best_first_neighbor_sorted_score_r' , 'best_first_neighbor_random_domination','best_first_neighbor_sorted_domination'}

# umda hyper-parameters #
selection_scheme = ['nds']  # {'nds','monoscore'}
replacement_scheme = ['elitism']  # {'elitism','replacement'}

# pbil hyper-parameters #
learning_rate = [0.1]
mutation_prob_pbil = [0.1]
mutation_shift = [0.1]

# feda hyper-parameters #
selection_scheme_feda = ['nds']  # {'nds','monoscore'}

''' returns a list of uid files created from geneticNDS hyper-parameters '''


def get_geneticnds_uids() -> [str]:
    uids_list = []

    for dependency in dependencies:
        for data in dataset:
            for max_evals in max_evaluations:
                for pop_size in population_size:
                    for iterations in num_iterations:
                        for sel in selection:
                            for xover in crossover:
                                for candidates in selection_candidates:
                                    for xover_prob in crossover_prob:
                                        for mut_prob in mutation_prob:
                                            for mut in mutation:
                                                for rep in replacement:
                                                    uid = '' + 'geneticNDS' + dependency + data + str(seed) + \
                                                          str(pop_size) + str(iterations) + str(max_evals) + \
                                                          str(candidates) + str(xover_prob) + str(mut_prob) + \
                                                          sel + xover + mut + rep + str(num_executions)
                                                    print(uid)
                                                    uids_list.append(uid)
    return uids_list


''' get_grasp_uids returns a list of uid files created from GRASP hyper-parameters '''


def get_grasp_uids() -> [str]:
    uids_list = []

    for dependency in dependencies:
        for data in dataset:
            for max_evals in max_evaluations:
                for pop_size in population_size:
                    for iterations in num_iterations:
                        for init in init_type:
                            for pr in path_relinking_mode:
                                for search in local_search_type:
                                    uid = '' + 'GRASP' + dependency + data + str(seed) + str(iterations) + \
                                          str(pop_size) + str(max_evals) + init + search + pr + str(num_executions)
                                    print(uid)
                                    uids_list.append(uid)
    return uids_list


''' returns a list of uid files created from umda hyper-parameters '''


def get_umda_uids() -> [str]:
    uids_list = []

    for dependency in dependencies:
        for data in dataset:
            for max_evals in max_evaluations:
                for pop_size in population_size:
                    for iterations in num_iterations:
                        for sel_scheme in selection_scheme:
                            for rep_scheme in replacement_scheme:
                                uid = '' + 'umda' + dependency + data + str(seed) + str(pop_size) + str(iterations) + \
                                      str(max_evals) + sel_scheme + rep_scheme + str(num_executions)
                                print(uid)
                                uids_list.append(uid)
    return uids_list


''' returns a list of uid files created from pbil hyper-parameters '''


def get_pbil_uids() -> [str]:
    uids_list = []

    for dependency in dependencies:
        for data in dataset:
            for max_evals in max_evaluations:
                for pop_size in population_size:
                    for iterations in num_iterations:
                        for l_rate in learning_rate:
                            for mut_prob_pbil in mutation_prob_pbil:
                                for shift in mutation_shift:
                                    uid = '' + 'pbil' + dependency + data + str(seed) + str(pop_size) + \
                                          str(iterations) + str(max_evals) + str(l_rate) + str(mut_prob_pbil) + \
                                          str(shift) + str(num_executions)
                                    print(uid)
                                    uids_list.append(uid)
    return uids_list


''' returns a list of uid files created from feda hyper-parameters '''


def get_feda_uids() -> [str]:
    uids_list = []

    for dependency in dependencies:
        for data in dataset:
            for max_evals in max_evaluations:
                for pop_size in population_size:
                    for iterations in num_iterations:
                        for sel_scheme in selection_scheme_feda:
                            uid = '' + 'umda' + dependency + data + str(seed) + str(pop_size) + str(iterations) + \
                                  str(max_evals) + sel_scheme + str(num_executions)
                            print(uid)
                            uids_list.append(uid)
    return uids_list


if __name__ == '__main__':
    files_uid = []

    if 'GRASP' in algorithms:
        files_uid = files_uid + get_grasp_uids()
    if 'geneticNDS' in algorithms:
        files_uid = files_uid + get_geneticnds_uids()
    if 'umda' in algorithms:
        files_uid = files_uid + get_umda_uids()
    if 'pbil' in algorithms:
        files_uid = files_uid + get_pbil_uids()
    if 'feda' in algorithms:
        files_uid = files_uid + get_feda_uids()

    print(f"\nGD+ and UNFR will be calculated using as reference the best pareto found in the {len(files_uid)}"
          f" files above.")
