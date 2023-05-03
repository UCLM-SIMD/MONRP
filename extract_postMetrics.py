import copy
import json
import string

from evaluation.get_nondominated_solutions import get_nondominated_solutions
from evaluation.metrics import calculate_gdplus, calculate_unfr
from models.Solution import Solution

""" Please fill the experiments hyper-parameters for all algorithms, which will be used to define the which results
will be taken into account to find the reference Pareto for GD+ and UNFR"""

# prefix = 'files_list_allGRASP_D'
prefix = 'files_list_FEASFIRST_'  # FEASFIRST

# agemoea2 and ctaea
repair_deps = [True]  # [True, False] # False for Feasibiliy first. true for repair per iteration

dependencies = ['True']  # {'True','False'}

# post metrics are not computed among results for all indicated datasets.Only 1 dataset is taken into account each time.
# dX files are classic (like cX files) but with a larger number of implied pbis by dependency and larger number of pbis
# do not use c5 and c6 because with 500 pbis its too slow
# p1', 'p2', 'a1', 'a2', 'a3', 'a4', 'c1', 'c2', 'c3', 'c4', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7'
# 'p1', 'p2', 's1','s2','s3','s4'
dataset = ['p1', 'p2', 'a1', 'a2', 'a3', 'a4', 'c1', 'c2', 'c3', 'c4', 'd1', 'd2', 'd3', 'd4']
# 'agemoea2', 'umda', 'pbil', 'GRASP', 'geneticnds', 'mimic','nsgaii', 'ctaea'
algorithm = ['agemoea2', 'geneticnds', 'umda', 'pbil', 'mimic', 'ctaea', 'feda']

# COMMON HYPER-PARAMETERS #
# possible algorithm values: {'GRASP', 'feda', 'geneticnds', 'pbil', 'umda', 'mimic''}
seed = 5
num_executions = 30  # 30 # 10 30
subset_size = [10]  # number of solutions to choose from final NDS in each algorithm to compute metrics
sss_type = [0]  # 0 for greedy hv
sss_per_iteration = [False]  # [True] # [True, False]
population_size = [100, 200, 500, 700, 1000]  # 2000, 3000] # 2000 and 3000 not in nsgaii (too slow)
num_generations = [50, 100, 200, 300, 400]  # 500 and 600 not in nsgaii
max_evals = [0]

# geneticNDS and NSGAii hyperparameters #
selection_candidates = [2]
crossover_prob = [0.8]
mutation_prob = [0.1]  # [0.1, 0.3]
mutation = ['flip1bit']  # {'flip1bit', 'flipeachbit'}
replacement = ['elitismnds']  # {'elitism', 'elitismnds'}
selection = ['tournament']  # only 'tournament' available
crossover = ['onepoint']  # only 'onepoint' available

# GRASP hyper-parameters #
init_type = ['stochastically']  # {'stochastically', 'uniform'}
path_relinking_mode = ['None', 'after_local']  # {'None', 'after_local'}
local_search_type = ['None', 'best_first_neighbor_random', 'best_first_neighbor_sorted_score',
                     'best_first_neighbor_sorted_score_r', 'best_first_neighbor_random_domination',
                     'best_first_neighbor_sorted_domination',
                     'local_search_bitwise_bestFirst_HV']
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

# mimic hyper-parameters #
selection_scheme_mimic = ['nds']  # {'nds','monoscore'}
rep_scheme_mimic = ["replacement"]  # actually, never used inside algorithm.
selected_individuals = [50]

''' returns a list of uid files created from geneticNDS hyper-parameters '''


def get_genetic_uids(name: str, d: str) -> [str]:
    uids_list = []

    for dependency in dependencies:
        for max_evalu in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for sel in selection:
                        for xover in crossover:
                            for candidates in selection_candidates:
                                for xover_prob in crossover_prob:
                                    for mut_prob in mutation_prob:
                                        for mut in mutation:
                                            for rep in replacement:
                                                rep = 'elitism' if name in ['nsgaii', 'nsgaiipt'] and \
                                                                   rep == 'elitismnds' else rep
                                                for size in subset_size:
                                                    for s_type in sss_type:
                                                        for s_per_it in sss_per_iteration:
                                                            uid_genetic = output_folder + name + dependency + d + \
                                                                          str(seed) + str(size) + str(s_type) + \
                                                                          str(s_per_it) + str(pop_size) + \
                                                                          str(iterations) + str(max_evalu) + \
                                                                          str(candidates) + str(xover_prob) + str(
                                                                mut_prob) + \
                                                                          sel + xover + mut + rep + str(
                                                                num_executions) + \
                                                                          '.json'
                                                            if (name in ['nsgaii',
                                                                         'nsgaiipt']):  # nsga is not compatible with SSS per iteration
                                                                uid_genetic = output_folder + name + dependency + d + \
                                                                              str(seed) + str(size) + str(s_type) + \
                                                                              'False' + str(pop_size) + \
                                                                              str(iterations) + str(max_evalu) + \
                                                                              str(candidates) + str(xover_prob) + str(
                                                                    mut_prob) + sel + xover + mut + rep + str(
                                                                    num_executions) + \
                                                                              '.json'
                                                            print('\'../' + uid_genetic + '\',')
                                                            uids_list.append(uid_genetic)
    return uids_list


def get_agemoea_uids(d: str) -> [str]:
    uids_list = []

    for pop_size in population_size:
        for iterations in num_generations:
            for size in subset_size:
                for s_type in sss_type:
                    for repair_D in repair_deps:
                        uid_agemoea = output_folder + 'agemoea2' + 'True' + d + \
                                      str(seed) + str(size) + str(s_type) + \
                                      'False' + str(pop_size) + \
                                      str(iterations) + '0' + str(repair_D) + str(num_executions) + \
                                      '.json'

                        print('\'../' + uid_agemoea + '\',')
                        uids_list.append(uid_agemoea)
    return uids_list


def get_ctaea_uids(d: str) -> [str]:
    uids_list = []

    for pop_size in population_size:
        for iterations in num_generations:
            for size in subset_size:
                for s_type in sss_type:
                    for repair_D in repair_deps:
                        uid_ctaea = output_folder + 'ctaea' + 'True' + d + \
                                      str(seed) + str(size) + str(s_type) + \
                                      'False' + str(pop_size) + \
                                      str(iterations) + '0' + str(repair_D) + str(num_executions) + \
                                      '.json'

                        print('\'../' + uid_ctaea + '\',')
                        uids_list.append(uid_ctaea)
    return uids_list


''' get_grasp_uids returns a list of uid files created from GRASP hyper-parameters '''


def get_grasp_uids(d: str) -> [str]:
    uids_list = []

    for dependency in dependencies:

        for max_evalu in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for init in init_type:
                        for pr in path_relinking_mode:
                            for search in local_search_type:
                                for size in subset_size:
                                    for s_type in sss_type:
                                        for s_per_it in sss_per_iteration:
                                            uid_grasp = output_folder + 'GRASP' + dependency + d + str(seed) + str(
                                                size) + \
                                                        str(s_type) + str(s_per_it) + str(iterations) + str(pop_size) + \
                                                        str(max_evalu) + init + search + pr + str(
                                                num_executions) + '.json'
                                            print('\'../' + uid_grasp + '\',')
                                            uids_list.append(uid_grasp)
    return uids_list


''' returns a list of uid files created from umda hyper-parameters '''


def get_umda_uids(d: str) -> [str]:
    uids_list = []

    for dependency in dependencies:

        for max_evalu in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for sel_scheme in selection_scheme:
                        for rep_scheme in replacement_scheme:
                            for size in subset_size:
                                for s_type in sss_type:
                                    for s_per_it in sss_per_iteration:
                                        uid_umda = output_folder + 'umda' + dependency + d + str(seed) + str(size) + \
                                                   str(s_type) + str(s_per_it) + \
                                                   str(pop_size) + str(iterations) + str(max_evalu) + sel_scheme + \
                                                   rep_scheme + str(num_executions) + '.json'
                                        print('\'../' + uid_umda + '\',')
                                        uids_list.append(uid_umda)
    return uids_list


''' returns a list of uid files created from random hyper-parameters '''


def get_random_uids(d: str) -> [str]:
    uids_list = []

    for dependency in dependencies:

        for max_evalu in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for size in subset_size:
                        for s_type in sss_type:
                            for s_per_it in sss_per_iteration:
                                uid_random = output_folder + 'random' + dependency + d + str(seed) + str(size) + \
                                             str(s_type) + str(s_per_it) + \
                                             str(pop_size) + str(iterations) + str(max_evalu) \
                                             + str(num_executions) + '.json'
                                print('\'../' + uid_random + '\',')
                                uids_list.append(uid_random)
    return uids_list


''' returns a list of uid files created from pbil hyper-parameters '''


def get_pbil_uids(d: str) -> [str]:
    uids_list = []

    for dependency in dependencies:

        for max_evalu in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for l_rate in learning_rate:
                        for mut_prob_pbil in mutation_prob_pbil:
                            for shift in mutation_shift:
                                for size in subset_size:
                                    for s_type in sss_type:
                                        for s_per_it in sss_per_iteration:
                                            uid_pbil = output_folder + 'pbil' + dependency + d + str(seed) + str(size) + \
                                                       str(s_type) + str(s_per_it) + str(pop_size) + \
                                                       str(iterations) + str(max_evalu) + str(l_rate) + str(
                                                mut_prob_pbil) + \
                                                       str(shift) + str(num_executions) + '.json'
                                            print('\'../' + uid_pbil + '\',')
                                            uids_list.append(uid_pbil)
    return uids_list


''' returns a list of uid files created from mimiv hyper-parameters '''


def get_mimic_uids(d: str) -> [str]:
    uids_list = []

    for dependency in dependencies:

        for max_evalu in max_evals:
            for sel_ind in selected_individuals:
                for pop_size in population_size:
                    for iterations in num_generations:
                        for sel_scheme in selection_scheme_mimic:
                            for rep_scheme in rep_scheme_mimic:
                                for size in subset_size:
                                    for s_type in sss_type:
                                        for s_per_it in sss_per_iteration:
                                            uid_umda = output_folder + 'mimic' + dependency + d + str(seed) + str(
                                                size) + \
                                                       str(s_type) + str(s_per_it) + str(pop_size) + str(iterations) + \
                                                       str(max_evalu) + str(sel_ind) + sel_scheme + \
                                                       rep_scheme + str(num_executions) + '.json'
                                            print('\'../' + uid_umda + '\',')
                                            uids_list.append(uid_umda)
    return uids_list


''' returns a list of uid files created from feda hyper-parameters '''


def get_feda_uids(d: str) -> [str]:
    uids_list = []

    for dependency in dependencies:

        for max_evalu in max_evals:
            for pop_size in population_size:
                for iterations in num_generations:
                    for sel_scheme in selection_scheme_feda:
                        for size in subset_size:
                            for s_type in sss_type:
                                for s_per_it in sss_per_iteration:
                                    uid_feda = output_folder + 'feda' + dependency + d + str(seed) + str(size) + \
                                               str(s_type) + str(s_per_it) + str(pop_size) + \
                                               str(iterations) + str(max_evalu) + sel_scheme + str(
                                        num_executions) + '.json'

                                    print('\'../' + uid_feda + '\',')
                                    uids_list.append(uid_feda)
    return uids_list


def construct_store_reference_pareto(uids):
    # construct reference Pareto, to be used to calculate UNFR and GD+. It is constructed with the non dominated
    # solutions among all nds together from all experiments with hyperparameters depicted above

    # construct reference pareto front
    all_solutions = []
    updated_uids = copy.deepcopy(uids)
    for file in uids:
        try:
            with open(file, 'r') as f_temp:

                dictio = json.load(f_temp)

                paretos = dictio['paretos']
                for pareto in paretos:
                    for xy in pareto:
                        sol = Solution(dataset=None, probabilities=None, uniform=None,
                                       cost=xy[0], satisfaction=xy[1])
                        all_solutions.append(sol)
        except (FileNotFoundError, IOError):
            updated_uids.remove(file)
            print("File not found so not used to extract metrics: ", file)

    uids = copy.deepcopy(updated_uids)
    nds = get_nondominated_solutions(solutions=all_solutions)
    # print(f"Reference Pareto contains {len(nds)} solutions.")
    pareto = []
    for sol in nds:
        pareto.append([sol.total_cost, sol.total_satisfaction])

    # store reference pareto front
    for file in uids:
        try:
            with open(file, 'r') as f_temp:
                dictio = json.load(f_temp)
                dictio['Reference_Pareto'] = pareto
            with open(file, 'w') as f_temp:
                json.dump(dictio, f_temp, ensure_ascii=False, indent=4)
        except (FileNotFoundError, IOError):
            pass

    return pareto, uids


# for each pareto in each file in files_uid, compute and store gd_plus respect to reference pareto front
def compute_and_store_gdplus(rpf, uids):
    for file in uids:
        try:
            with open(file, 'r') as file_without_gd:
                dictio = json.load(file_without_gd)
                paretos = dictio['paretos']
                gd_plus = []
                for pareto in paretos:
                    gd_plus.insert(len(gd_plus), calculate_gdplus(pareto, rpf))
                    metrics = dictio['metrics']
                metrics['gdplus'] = gd_plus
            with open(file, 'w') as file_with_gd:
                json.dump(dictio, file_with_gd, ensure_ascii=False, indent=4)
        except (FileNotFoundError, IOError):
            pass


# for each pareto in each file in files_uid, compute and store unfr respect to reference pareto front
def compute_and_store_unfr(rpf, uids):
    for file in uids:
        try:
            with open(file, 'r') as f_temp:
                dictio = json.load(f_temp)
                paretos = dictio['paretos']
                unfr = []
                for pareto in paretos:
                    unfr.insert(len(unfr), calculate_unfr(pareto, rpf))
                    metrics = dictio['metrics']
                metrics['unfr'] = unfr
            with open(file, 'w') as f_temp:
                json.dump(dictio, f_temp, ensure_ascii=False, indent=4)
        except (FileNotFoundError, IOError):
            pass


if __name__ == '__main__':
    print('GD+ and UNFR will be calculated using as reference the best pareto found in the output files given + \
                 the hyperparameters, for each dataset.')
    # folder with output files from which to extract
    all_files_uid = []
    for data in dataset:

        files_uid = []
        # find unique file ids from the list of hyperparameters set at the beginning of this file, above.
        if 'GRASP' in algorithm:
            output_folder = 'output/GRASP/'
            files_uid = files_uid + get_grasp_uids(data)
        if 'geneticnds' in algorithm:
            output_folder = 'output/geneticnds/'
            files_uid = files_uid + get_genetic_uids('geneticNDS', data)
        if 'nsgaii' in algorithm:
            output_folder = 'output/nsgaii/'
            files_uid = files_uid + get_genetic_uids('nsgaii', data)
        if 'nsgaiipt10to30' in algorithm:
            output_folder = 'output/nsgaiipt10to30/'
            files_uid = files_uid + get_genetic_uids('nsgaiipt', data)
        if 'nsgaiipt' in algorithm:
            output_folder = 'output/nsgaiipt/'
            files_uid = files_uid + get_genetic_uids('nsgaiipt', data)
        if 'umda' in algorithm:
            output_folder = 'output/umda/'
            files_uid = files_uid + get_umda_uids(data)
        if 'pbil' in algorithm:
            output_folder = 'output/pbil/'
            files_uid = files_uid + get_pbil_uids(data)
        if 'mimic' in algorithm:
            output_folder = 'output/mimic/'
            files_uid = files_uid + get_mimic_uids(data)
        if 'feda' in algorithm:
            output_folder = 'output/feda/'
            files_uid = files_uid + get_feda_uids(data)
        if 'agemoea2' in algorithm:
            output_folder = 'output/agemoea2/'
            files_uid = files_uid + get_agemoea_uids(data)
        if 'ctaea' in algorithm:
            output_folder = 'output/ctaea/'
            files_uid = files_uid + get_ctaea_uids(data)
        if 'random' in algorithm:
            output_folder = 'output/random/'
            files_uid = files_uid + get_random_uids(data)

        # find Reference Pareto and compute metrics, and remove files not available yet
        reference_pareto, files_uid = construct_store_reference_pareto(files_uid)

        compute_and_store_gdplus(rpf=reference_pareto, uids=files_uid)
        compute_and_store_unfr(rpf=reference_pareto, uids=files_uid)

        print()
        for f in files_uid:
            all_files_uid.append(f)

    # store all uids in container file (for use in analysis jupyter notebook)
    sufix = ''
    for alg in algorithm:
        sufix += alg + '-'
    container_name = prefix + sufix[0:len(sufix) - 1]
    with open('output/' + container_name, 'w') as container_file:
        for uid in all_files_uid:
            container_file.write(uid + "\n")
    print(f"File list {container_name} created to be used from analisis notebook")
