dependencies = ['D']  # {'D', 'd'}
# do not use c5 and c6 because with 500 pbis its too slow
dataset =  ['p1', 'p2', 'a1', 'a2', 'a3', 'a4', 'c1', 'c2', 'c3', 'c4', 'd1', 'd2', 'd3','d4','d5','d6','d7']


# COMMON HYPER-PARAMETERS #
# possible algorithm values: {'GRASP', 'feda', 'geneticnds', 'pbil', 'umda', nsgaii}
algorithm = 'GRASP'  # 'GRASP', 'geneticnds', 'nsgaii', 'umda', 'pbil', 'feda'
seed = 5
num_executions = 30
subset_size = [10]  # number of solutions to choose from final NDS in each algorithm to compute metrics
population_size = [100, 200, 500, 700, 1000]
num_generations = [50, 100, 200, 300, 400]
max_evals = [0]

# geneticNDS and NSGAii hyperparameters #
selection_candidates = [2]
crossover_prob = [0.8]
mutation_prob = [0.1, 0.3]  # [0.1, 0.3]
mutation = ['flip1bit']  # {'flip1bit', 'flipeachbit'}
replacement = ['elitismnds']  # {'elitism', 'elitismnds'}
selection = ['tournament']  # only 'tournament' available
crossover = ['onepoint']  # only 'onepoint' available

# GRASP hyper-parameters #
init_type = ['stochastically']  # {'stochastically', 'uniform'}
path_relinking_mode = ['None', 'PR']  # {'None', 'PR'}
local_search_type = ['best_first_neighbor_random_domination'] #['best_first_neighbor_random',best_first_neighbor_random_domination']
# {'None', 'best_first_neighbor_random',
            # 'best_first_neighbor_sorted_score', best_first_neighbor_sorted_score_r' ,
            # 'best_first_neighbor_random_domination','best_first_neighbor_sorted_domination'}


# umda hyper-parameters #
selection_scheme = ['nds']  # {'nds','monoscore'}
replacement_scheme = ['elitism']  # {'elitism','replacement'}

# pbil hyper-parameters #
learning_rate = [0.1]
mutation_prob_pbil = [0.1]
mutation_shift = [0.1]

# feda hyper-parameters #
selection_scheme_feda = ['nds']  # {'nds','monoscore'}


def get_genetic_options(name: str, dataset_name: str) -> [str]:
    params_list = []

    for dependency in dependencies:
        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for sel in selection:
                        for xover in crossover:
                            for candidates in selection_candidates:
                                for xover_prob in crossover_prob:
                                    for mut_prob in mutation_prob:
                                        for mut in mutation:
                                            for rep in replacement:
                                                for size in subset_size:
                                                    params_line = f"genetic {name} {dataset_name} {str(seed)} " \
                                                                  f"{str(pop_size)}" \
                                                                  f" {str(iterations)} {str(max_evaluations)} {sel} " \
                                                                  f"{candidates} {xover} {str(xover_prob)} {mut} " \
                                                                  f"{str(mut_prob)} {rep} {str(num_executions)} " \
                                                                  f"{dependency} {str(size)}"

                                                    params_list.append(params_line)
    return params_list


def get_grasp_options(dataset_name: str) -> [str]:
    params_list = []

    for dependency in dependencies:
        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for init in init_type:
                        for pr in path_relinking_mode:
                            for search in local_search_type:
                                for size in subset_size:
                                    params_line = f"grasp grasp {dataset_name} {str(seed)} {str(iterations)} " \
                                                  f"{str(pop_size)} {str(max_evaluations)} {init} {search}" \
                                                  f" {pr} {str(num_executions)} {dependency} {str(size)}"
                                    params_list.append(params_line)
    return params_list


def get_umda_options(dataset_name: str) -> [str]:
    params_list = []

    for dependency in dependencies:

        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for sel_scheme in selection_scheme:
                        for rep_scheme in replacement_scheme:
                            for size in subset_size:
                                params_line = f"eda umda {dataset_name} {str(seed)} {str(pop_size)}" \
                                              f" {str(iterations)} " \
                                              f"{str(max_evaluations)} 2 {sel_scheme} {rep_scheme} " \
                                              f"{str(num_executions)} {dependency} {str(size)}"

                                params_list.append(params_line)
    return params_list


def get_pbil_options(dataset_name: str) -> [str]:
    params_list = []

    for dependency in dependencies:

        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for l_rate in learning_rate:
                        for mut_prob_pbil in mutation_prob_pbil:
                            for shift in mutation_shift:
                                for size in subset_size:
                                    params_line = f"eda pbil {dataset_name} {str(seed)} {str(pop_size)} " \
                                                  f"{str(iterations)} " \
                                                  f"{str(max_evaluations)} {str(l_rate)} {str(mut_prob_pbil)}" \
                                                  f" {str(shift)} " \
                                                  f"{str(num_executions)} {dependency} {str(size)}"

                                    params_list.append(params_line)
    return params_list


def get_feda_options(dataset_name: str) -> [str]:
    params_list = []

    for dependency in dependencies:

        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for sel_scheme in selection_scheme_feda:
                        for size in subset_size:
                            params_line = f"eda feda {dataset_name} {str(seed)} {str(pop_size)} {str(iterations)} " \
                                          f"{str(max_evaluations)} " \
                                          f"{str(num_executions)} {dependency} {str(size)} {sel_scheme}"

                            params_list.append(params_line)
    return params_list


"""""""""""""CREATE PARAMETERS (OPTIONS) FILES """""""""""""
options_list = []

for data in dataset:

    if 'GRASP' == algorithm:
        options_list = options_list + get_grasp_options(data)
    if 'geneticnds' == algorithm:
        options_list = options_list + get_genetic_options('geneticnds', data)
    if 'nsgaii' == algorithm:
        options_list = options_list + get_genetic_options('nsgaii', data)
    if 'umda' == algorithm:
        options_list = options_list + get_umda_options(data)
    if 'pbil' == algorithm:
        options_list = options_list + get_pbil_options(data)
    if 'feda' == algorithm:
        options_list = options_list + get_feda_options(data)

params_file = open("pablo/params_file", "w", newline='\n')
for line in options_list:
    params_file.write(line + "\n")
params_file.close()

""""""""""" create PBS file for running experiments """""""""""
njobs = options_list.__len__()
if njobs == 1:
    njobs = 2

pbs = f"#!/bin/sh \n #PBS -N {algorithm} \n #PBS -l mem=2500mb \n #PBS -J 1-{str(njobs)} \n"

template = open("pablo/template", 'r').read()
pbs_file = open("pablo/runJobs.pbs", 'w', newline='\n')  # LF format for galgo
pbs_file.write(pbs + template)
pbs_file.close()
