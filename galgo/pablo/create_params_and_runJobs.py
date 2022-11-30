<<<<<<< HEAD
<<<<<<< HEAD
dependencies = ['D']  # {'D', 'd'}
# ['p1', 'p2', 'a1', 'a2', 'a3', 'a4', 'c1', 'c2', 'c3', 'c4', 'd1', 'd2', 'd3','d4','d5','d6','d7',
#'e1', 'e2', 'e3','e4','e5','e6']
dataset = ['p1', 'p2', 's1','s2','s3']

# COMMON HYPER-PARAMETERS #
# possible algorithm values: {'GRASP', 'feda', 'geneticnds', 'pbil', 'umda', nsgaii}
algorithm = 'feda'  # 'GRASP', 'geneticnds', 'nsgaii', 'umda', 'pbil', 'feda', 'mimic'
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
=======

import datetime
=======
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
dependencies = ['D']  # {'D', 'd'}
# ['p1', 'p2', 'a1', 'a2', 'a3', 'a4', 'c1', 'c2', 'c3', 'c4', 'd1', 'd2', 'd3','d4','d5','d6','d7',
#'e1', 'e2', 'e3','e4','e5','e6']
dataset = ['d1', 'd2', 'd3', 'd4']

# COMMON HYPER-PARAMETERS #
# possible algorithm values: {'GRASP', 'feda', 'geneticnds', 'pbil', 'umda', nsgaii}
algorithm = 'mimic'  # 'GRASP', 'geneticnds', 'nsgaii', 'umda', 'pbil', 'feda', 'mimic'
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
<<<<<<< HEAD
replacement = ['elitism']  # {'elitism', 'elitismnds'}
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======
replacement = ['elitismnds']  # {'elitism', 'elitismnds'}
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
selection = ['tournament']  # only 'tournament' available
crossover = ['onepoint']  # only 'onepoint' available

# GRASP hyper-parameters #
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
init_type = ['stochastically']  # {'stochastically', 'uniform'}
path_relinking_mode = ['None', 'PR']  # {'None', 'PR'}
local_search_type = [
    'best_first_neighbor_random_domination']  # ['best_first_neighbor_random',best_first_neighbor_random_domination']
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

# mimic hyper-parameters #
selection_scheme_mimic = ['nds']  # {'nds','monoscore'}
rep_scheme_mimic = ["replacement"]  # actually, never used inside algorithm.
selected_individuals = [50, 100]


def get_genetic_options(name: str, dataset_name: str) -> [str]:
    params_list = []

    for dependency in dependencies:
        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
=======
max_evals_grasp = [10000]
=======
max_evals_grasp = [0] # stop criteria is grasp_iterations
>>>>>>> a1359f27 (solved issue when comparing new solutions to nds (.isclose). now solution subset search has a better general ref point.)
=======
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
init_type = ['stochastically']  # {'stochastically', 'uniform'}
path_relinking_mode = ['None', 'PR']  # {'None', 'PR'}
local_search_type = [
    'best_first_neighbor_random_domination']  # ['best_first_neighbor_random',best_first_neighbor_random_domination']
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

# mimic hyper-parameters #
selection_scheme_mimic = ['nds']  # {'nds','monoscore'}
rep_scheme_mimic = ["replacement"]  # actually, never used inside algorithm.
selected_individuals = [50, 100]


def get_genetic_options(name: str, dataset_name: str) -> [str]:
    params_list = []

    for dependency in dependencies:
        for max_evaluations in max_evals:
            for pop_size in population_size:
<<<<<<< HEAD
                for iterations in num_iterations:
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======
                for iterations in num_generations:
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
                    for sel in selection:
                        for xover in crossover:
                            for candidates in selection_candidates:
                                for xover_prob in crossover_prob:
                                    for mut_prob in mutation_prob:
                                        for mut in mutation:
                                            for rep in replacement:
                                                for size in subset_size:
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
                                                    params_line = f"genetic {name} {dataset_name} {str(seed)} " \
                                                                  f"{str(pop_size)}" \
                                                                  f" {str(iterations)} {str(max_evaluations)} {sel} " \
                                                                  f"{candidates} {xover} {str(xover_prob)} {mut} " \
<<<<<<< HEAD
=======
                                                    params_line = f"genetic {name} {data} {str(seed)} {str(pop_size)}" \
                                                                  f" {str(iterations)} {str(max_evals)} {sel} " \
                                                                  f"{candidates} {xover} {str(xover_prob)} {mut} "\
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
                                                                  f"{str(mut_prob)} {rep} {str(num_executions)} " \
                                                                  f"{dependency} {str(size)}"

                                                    params_list.append(params_line)
    return params_list


<<<<<<< HEAD
<<<<<<< HEAD
def get_grasp_options(dataset_name: str) -> [str]:
    params_list = []

    for dependency in dependencies:
        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
=======
def get_grasp_options(data: str) -> [str]:
    params_list = []

    for dependency in dependencies:
      for max_evals in max_evals_grasp:
            for pop_size in solutions_per_iteration:
                for iterations in grasp_iterations:
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======
def get_grasp_options(dataset_name: str) -> [str]:
    params_list = []

    for dependency in dependencies:
        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
                    for init in init_type:
                        for pr in path_relinking_mode:
                            for search in local_search_type:
                                for size in subset_size:
<<<<<<< HEAD
<<<<<<< HEAD
                                    params_line = f"grasp grasp {dataset_name} {str(seed)} {str(iterations)} " \
                                                  f"{str(pop_size)} {str(max_evaluations)} {init} {search}" \
=======
                                    params_line = f"grasp grasp {data} {str(seed)} {str(iterations)} " \
                                                  f"{str(pop_size)} {str(max_evals)} {init} {search}" \
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======
                                    params_line = f"grasp grasp {dataset_name} {str(seed)} {str(iterations)} " \
                                                  f"{str(pop_size)} {str(max_evaluations)} {init} {search}" \
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
                                                  f" {pr} {str(num_executions)} {dependency} {str(size)}"
                                    params_list.append(params_line)
    return params_list

<<<<<<< HEAD
<<<<<<< HEAD

def get_umda_options(dataset_name: str) -> [str]:
=======
def get_umda_options(data: str) -> [str]:
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======

def get_umda_options(dataset_name: str) -> [str]:
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
    params_list = []

    for dependency in dependencies:

<<<<<<< HEAD
<<<<<<< HEAD
        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for sel_scheme in selection_scheme:
                        for rep_scheme in replacement_scheme:
                            for size in subset_size:
                                params_line = f"eda umda {dataset_name} {str(seed)} {str(pop_size)}" \
                                              f" {str(iterations)} " \
                                              f"{str(max_evaluations)} 2 {sel_scheme} {rep_scheme} " \
=======
        for max_evals in max_evals_umda:
            for pop_size in population_length_umda:
                for iterations in max_generations_umda:
                    for sel_scheme in selection_scheme:
                        for rep_scheme in replacement_scheme:
                            for size in subset_size:
                                params_line = f"eda umda {data} {str(seed)} {str(pop_size)} {str(iterations)}" \
                                              f"{str(max_evals)} 2 {sel_scheme} {rep_scheme}" \
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======
        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for sel_scheme in selection_scheme:
                        for rep_scheme in replacement_scheme:
                            for size in subset_size:
                                params_line = f"eda umda {dataset_name} {str(seed)} {str(pop_size)}" \
                                              f" {str(iterations)} " \
                                              f"{str(max_evaluations)} 2 {sel_scheme} {rep_scheme} " \
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
                                              f"{str(num_executions)} {dependency} {str(size)}"

                                params_list.append(params_line)
    return params_list


<<<<<<< HEAD
<<<<<<< HEAD
def get_pbil_options(dataset_name: str) -> [str]:
=======
def get_pbil_options(data: str) -> [str]:
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======
def get_pbil_options(dataset_name: str) -> [str]:
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
    params_list = []

    for dependency in dependencies:

<<<<<<< HEAD
<<<<<<< HEAD
        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
=======
        for max_evals in max_evals_pbil:
            for pop_size in population_length_pbil:
                for iterations in max_generations_pbil:
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======
        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
                    for l_rate in learning_rate:
                        for mut_prob_pbil in mutation_prob_pbil:
                            for shift in mutation_shift:
                                for size in subset_size:
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
                                    params_line = f"eda pbil {dataset_name} {str(seed)} {str(pop_size)} " \
                                                  f"{str(iterations)} " \
                                                  f"{str(max_evaluations)} {str(l_rate)} {str(mut_prob_pbil)}" \
                                                  f" {str(shift)} " \
<<<<<<< HEAD
=======

                                    params_line = f"eda pbil {data} {str(seed)} {str(pop_size)} {str(iterations)}" \
                                                  f"{str(max_evals)} {str(l_rate)} {str(mut_prob_pbil)} {str(shift)}" \
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
                                                  f"{str(num_executions)} {dependency} {str(size)}"

                                    params_list.append(params_line)
    return params_list


<<<<<<< HEAD
<<<<<<< HEAD
def get_feda_options(dataset_name: str) -> [str]:
=======
def get_feda_options(data: str) -> [str]:
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======
def get_feda_options(dataset_name: str) -> [str]:
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
    params_list = []

    for dependency in dependencies:

<<<<<<< HEAD
<<<<<<< HEAD
        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for sel_scheme in selection_scheme_feda:
                        for size in subset_size:
                            params_line = f"eda feda {dataset_name} {str(seed)} {str(pop_size)} {str(iterations)} " \
                                          f"{str(max_evaluations)} " \
                                          f"{str(num_executions)} {dependency} {str(size)} {sel_scheme}"
=======
        for max_evals in max_evals_feda:
            for pop_size in population_size_feda:
                for iterations in num_iterations_feda:
                    for sel_scheme in selection_scheme_feda:
                        for size in subset_size:
                            params_line = f"eda feda {data} {str(seed)} {str(pop_size)} {str(iterations)}" \
                                                  f"{str(max_evals)} " \
                                          f"{str(num_executions)} {dependency} {str(size)}"
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======
        for max_evaluations in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for sel_scheme in selection_scheme_feda:
                        for size in subset_size:
                            params_line = f"eda feda {dataset_name} {str(seed)} {str(pop_size)} {str(iterations)} " \
                                          f"{str(max_evaluations)} " \
                                          f"{str(num_executions)} {dependency} {str(size)} {sel_scheme}"
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)

                            params_list.append(params_line)
    return params_list

<<<<<<< HEAD
<<<<<<< HEAD

def get_mimic_options(dataset_name: str) -> [str]:
    params_list = []

    for pop_size in population_size:
        for iterations in num_generations:
            for max_evaluations in max_evals:
                for rep_scheme in rep_scheme_mimic:
                    for sel_individuals in selected_individuals:
                        for sel_scheme in selection_scheme_mimic:
                            for dependency in dependencies:
                                for size in subset_size:
                                    params_line = f"eda mimic {dataset_name} {str(seed)} {str(pop_size)}" \
                                                  f" {str(iterations)} " \
                                                  f"{str(max_evaluations)} {str(rep_scheme)} {int(sel_individuals)} " \
                                                  f"{str(sel_scheme)} {int(num_executions)} {dependency} {str(size)}"

                                    params_list.append(params_line)
    return params_list


<<<<<<< HEAD
=======
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======

>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
=======
>>>>>>> 80ff396a (mimic added to execution framework. swapping of unfr and gdplus in jupyter notebook solved.)
"""""""""""""CREATE PARAMETERS (OPTIONS) FILES """""""""""""
options_list = []

for data in dataset:

<<<<<<< HEAD
<<<<<<< HEAD
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
    if 'mimic' == algorithm:
        options_list = options_list + get_mimic_options(data)
<<<<<<< HEAD

params_file = open("pablo/params_file", "w", newline='\n')
for line in options_list:
    params_file.write(line + "\n")
=======
    if 'GRASP' in algorithms:
=======
    if 'GRASP' == algorithm:
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
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
=======
>>>>>>> 80ff396a (mimic added to execution framework. swapping of unfr and gdplus in jupyter notebook solved.)

params_file = open("pablo/params_file", "w", newline='\n')
for line in options_list:
<<<<<<< HEAD
    params_file.write(line+"\n")
>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======
    params_file.write(line + "\n")
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
params_file.close()

""""""""""" create PBS file for running experiments """""""""""
njobs = options_list.__len__()
if njobs == 1:
    njobs = 2

<<<<<<< HEAD
<<<<<<< HEAD
pbs = f"#!/bin/sh \n #PBS -N {algorithm} \n #PBS -l mem=2500mb \n #PBS -J 1-{str(njobs)} \n"

template = open("pablo/template", 'r').read()
pbs_file = open("pablo/runJobs.pbs", 'w', newline='\n')  # LF format for galgo
pbs_file.write(pbs + template)
pbs_file.close()
=======


pbs = '#!/bin/sh \n #PBS -N MONRP \n #PBS -l mem=2500mb \n #PBS -J 1-' + str(njobs) + '\n'
=======
pbs = f"#!/bin/sh \n #PBS -N {algorithm} \n #PBS -l mem=2500mb \n #PBS -J 1-{str(njobs)} \n"
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)

template = open("pablo/template", 'r').read()
pbs_file = open("pablo/runJobs.pbs", 'w', newline='\n')  # LF format for galgo
pbs_file.write(pbs + template)
pbs_file.close()
<<<<<<< HEAD


>>>>>>> facfd1a9 (galgo/pablo contains scripts to create params file and run jobs in galgo)
=======
>>>>>>> a20168f3 (refactoring en create_params_and_runJobs.py y arreglo de algunos bug en parametros en extract_postMetrics.py)
