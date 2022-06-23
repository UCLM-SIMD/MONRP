
import datetime
dependencies = ['D']  # {'D', 'd'}


dataset = ['p1','p2','a1','a2','a3','a4','c1','c2','c3','c4','c5','c6']  # {'p1','p2','a1','a2','a3','a4','c1','c2','c3','c4','c5','c6'}

# COMMON HYPER-PARAMETERS #
# possible algorithm values: {'GRASP', 'feda', 'geneticNDS', 'pbil', 'umda', nsgaii}
algorithms = ['GRASP']  # ['GRASP', 'geneticNDS', 'nsgaii', 'umda', 'pbil', 'feda']
seed = 5
num_executions = 30
subset_size = [10]  # number of solutions to choose from final NDS in each algorithm to compute metrics

# geneticNDS and NSGAii hyperparameters #
max_evals_genetic = [0]
selection_candidates = [2]
crossover_prob = [0.8]
mutation_prob = [0.1, 0.3]
population_size = [10, 20, 40]
num_iterations = [10]
mutation = ['flip1bit']  # {'flip1bit', 'flipeachbit'}
replacement = ['elitism']  # {'elitism', 'elitismnds'}
selection = ['tournament']  # only 'tournament' available
crossover = ['onepoint']  # only 'onepoint' available

# GRASP hyper-parameters #
max_evals_grasp = [0] # stop criteria is grasp_iterations
init_type = ['stochastically']  # {'stochastically', 'uniform'}
path_relinking_mode = ['None', 'PR']  # {'None', 'PR'}
local_search_type = ['best_first_neighbor_random']
grasp_iterations = [5, 10, 50, 100, 200, 300, 400]
solutions_per_iteration = [50, 100, 200, 500, 700, 1000]
# local_search_type values: {'None', 'best_first_neighbor_random','best_first_neighbor_sorted_score',
# best_first_neighbor_sorted_score_r' , 'best_first_neighbor_random_domination','best_first_neighbor_sorted_domination'}

# umda hyper-parameters #
max_evals_umda = [10000]
selection_scheme = ['nds']  # {'nds','monoscore'}
replacement_scheme = ['elitism']  # {'elitism','replacement'}
population_length_umda = [100]
max_generations_umda = [300]

# pbil hyper-parameters #
max_evals_pbil = [10000]
learning_rate = [0.1]
mutation_prob_pbil = [0.1]
mutation_shift = [0.1]
population_length_pbil = [100]
max_generations_pbil = [300]

# feda hyper-parameters #
max_evals_feda = [10000]
selection_scheme_feda = ['nds']  # {'nds','monoscore'}
population_size_feda = [100]
num_iterations_feda = [300]


def get_genetic_options(name: str, data: str) -> [str]:
    params_list = []

    for dependency in dependencies:
        for max_evals in max_evals_genetic:
            for pop_size in population_size:
                for iterations in num_iterations:
                    for sel in selection:
                        for xover in crossover:
                            for candidates in selection_candidates:
                                for xover_prob in crossover_prob:
                                    for mut_prob in mutation_prob:
                                        for mut in mutation:
                                            for rep in replacement:
                                                for size in subset_size:
                                                    params_line = f"genetic {name} {data} {str(seed)} {str(pop_size)}" \
                                                                  f" {str(iterations)} {str(max_evals)} {sel} " \
                                                                  f"{candidates} {xover} {str(xover_prob)} {mut} "\
                                                                  f"{str(mut_prob)} {rep} {str(num_executions)} " \
                                                                  f"{dependency} {str(size)}"

                                                    params_list.append(params_line)
    return params_list


def get_grasp_options(data: str) -> [str]:
    params_list = []

    for dependency in dependencies:
      for max_evals in max_evals_grasp:
            for pop_size in solutions_per_iteration:
                for iterations in grasp_iterations:
                    for init in init_type:
                        for pr in path_relinking_mode:
                            for search in local_search_type:
                                for size in subset_size:
                                    params_line = f"grasp grasp {data} {str(seed)} {str(iterations)} " \
                                                  f"{str(pop_size)} {str(max_evals)} {init} {search}" \
                                                  f" {pr} {str(num_executions)} {dependency} {str(size)}"
                                    params_list.append(params_line)
    return params_list

def get_umda_options(data: str) -> [str]:
    params_list = []

    for dependency in dependencies:

        for max_evals in max_evals_umda:
            for pop_size in population_length_umda:
                for iterations in max_generations_umda:
                    for sel_scheme in selection_scheme:
                        for rep_scheme in replacement_scheme:
                            for size in subset_size:
                                params_line = f"eda umda {data} {str(seed)} {str(pop_size)} {str(iterations)}" \
                                              f"{str(max_evals)} 2 {sel_scheme} {rep_scheme}" \
                                              f"{str(num_executions)} {dependency} {str(size)}"

                                params_list.append(params_line)
    return params_list


def get_pbil_options(data: str) -> [str]:
    params_list = []

    for dependency in dependencies:

        for max_evals in max_evals_pbil:
            for pop_size in population_length_pbil:
                for iterations in max_generations_pbil:
                    for l_rate in learning_rate:
                        for mut_prob_pbil in mutation_prob_pbil:
                            for shift in mutation_shift:
                                for size in subset_size:

                                    params_line = f"eda pbil {data} {str(seed)} {str(pop_size)} {str(iterations)}" \
                                                  f"{str(max_evals)} {str(l_rate)} {str(mut_prob_pbil)} {str(shift)}" \
                                                  f"{str(num_executions)} {dependency} {str(size)}"

                                    params_list.append(params_line)
    return params_list


def get_feda_options(data: str) -> [str]:
    params_list = []

    for dependency in dependencies:

        for max_evals in max_evals_feda:
            for pop_size in population_size_feda:
                for iterations in num_iterations_feda:
                    for sel_scheme in selection_scheme_feda:
                        for size in subset_size:
                            params_line = f"eda feda {data} {str(seed)} {str(pop_size)} {str(iterations)}" \
                                                  f"{str(max_evals)} " \
                                          f"{str(num_executions)} {dependency} {str(size)}"

                            params_list.append(params_line)
    return params_list

"""""""""""""CREATE PARAMETERS (OPTIONS) FILES """""""""""""
options_list = []

for data in dataset:

    if 'GRASP' in algorithms:
        options_list = options_list + get_grasp_options(data)
    if 'geneticNDS' in algorithms:
        options_list = options_list + get_genetic_options('geneticNDS', data)
    if 'nsgaii' in algorithms:
        options_list = options_list + get_genetic_options('nsgaii', data)
    if 'umda' in algorithms:
        options_list = options_list + get_umda_options(data)
    if 'pbil' in algorithms:
        options_list = options_list + get_pbil_options(data)
    if 'feda' in algorithms:
        options_list = options_list + get_feda_options(data)



params_file = open("pablo/params_file", "w",newline='\n')
for line in options_list:
    params_file.write(line+"\n")
params_file.close()

""""""""""" create PBS file for running experiments """""""""""
njobs = options_list.__len__()
if njobs == 1:
    njobs = 2



pbs = '#!/bin/sh \n #PBS -N MONRP \n #PBS -l mem=2500mb \n #PBS -J 1-' + str(njobs) + '\n'

template = open("pablo/template", 'r').read()
pbs_file = open("pablo/runJobs.pbs", 'w',newline='\n') #LF format for galgo
pbs_file.write(pbs + template)
pbs_file.close()


