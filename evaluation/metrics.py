import math
import numpy as np

# AVGVALUE------------------------------------------------------------------


def calculate_avgValue(population):
    avgValue = 0
    for ind in population:
        avgValue += ind.compute_mono_objective_score()
    avgValue /= len(population)
    return avgValue

# BESTAVGVALUE------------------------------------------------------------------


def calculate_bestAvgValue(population):
    bestAvgValue = 0
    for ind in population:
        if bestAvgValue < ind.compute_mono_objective_score():
            bestAvgValue = ind.compute_mono_objective_score()

    return bestAvgValue

# NUMSOLUTIONS------------------------------------------------------------------


def calculate_numSolutions(population):
    return len(set(population))

# SPACING------------------------------------------------------------------


def calculate_spacing(population):
    n = len(population)
    N = 2  # len(population[0].objectives)
    spacing = 0
    mean_objectives = []

    objective = 0
    for j in range(0, len(population)):
        objective += population[j].total_cost
    objective /= len(population)
    mean_objectives.append(objective)

    objective = 0
    for j in range(0, len(population)):
        objective += population[j].total_satisfaction
    objective /= len(population)
    mean_objectives.append(objective)

    # calcular la media de cada objetivo
    # for i in range(0, len(population[0].objectives)):
    #	objective = 0
    #	for j in range(0, len(population)):
    #		objective += population[j].objectives[i].value
    #	objective /= len(population)
    #	mean_objectives.append(objective)

    for j in range(0, len(population)):
        aux_spacing = 0
        for i in range(0, N):  # len(population[0].objectives)):
            di = mean_objectives[i]
            if i == 0:
                dij = population[j].total_cost
            elif i == 1:
                dij = population[j].total_satisfaction
            aux = (1 - (abs(dij) / di)) ** 2
            aux_spacing += aux
        aux_spacing = math.sqrt(aux_spacing)
        spacing += aux_spacing

    spacing /= (n * N)
    return spacing

# HYPERVOLUME------------------------------------------------------------------


def calculate_hypervolume(population):
    # obtener minimos y maximos de cada objetivo
    objectives_diff = []
    # aux_max_obj=[population[0].max_score,population[0].max_cost]
    # aux_min_obj=[population[0].min_score,population[0].min_cost]
    aux_max_cost, aux_max_sat = population[0].get_max_cost_satisfactions()
    aux_min_cost, aux_min_sat = population[0].get_min_cost_satisfactions()

    aux_min = float('inf')
    aux_max = 0
    for ind in population:
        if ind.total_cost < aux_min:
            aux_min = ind.total_cost
        if ind.total_cost > aux_max:
            aux_max = ind.total_cost
    aux_max_norm = (aux_max-aux_min_cost)/(aux_max_cost-aux_min_cost)
    aux_min_norm = (aux_min-aux_min_cost)/(aux_max_cost-aux_min_cost)
    aux_val = aux_max_norm-aux_min_norm
    objectives_diff.append(aux_val)

    aux_min = float('inf')
    aux_max = 0
    for ind in population:
        if ind.total_satisfaction < aux_min:
            aux_min = ind.total_satisfaction
        if ind.total_satisfaction > aux_max:
            aux_max = ind.total_satisfaction
    aux_max_norm = (aux_max-aux_min_sat)/(aux_max_sat-aux_min_sat)
    aux_min_norm = (aux_min-aux_min_sat)/(aux_max_sat-aux_min_sat)
    aux_val = aux_max_norm-aux_min_norm
    objectives_diff.append(aux_val)

    # for i in range(0,len(population[0].objectives)):
    #	aux_min = float('inf')
    #	aux_max = 0
    #	for ind in population:  ##############################################
    #		if ind.objectives[i].value < aux_min:
    #			aux_min = ind.objectives[i].value
    #		if ind.objectives[i].value > aux_max:
    #			aux_max = ind.objectives[i].value
#
    #	aux_max_norm = (aux_max-aux_min_obj[i])/(aux_max_obj[i]-aux_min_obj[i])
    #	aux_min_norm = (aux_min-aux_min_obj[i])/(aux_max_obj[i]-aux_min_obj[i])
    #	aux_val = aux_max_norm-aux_min_norm
    #	objectives_diff.append(aux_val)

    # calcular hypervolume
    hypervolume = 1
    for i in range(0, len(objectives_diff)):
        hypervolume *= objectives_diff[i]

    return hypervolume

# SPREAD------------------------------------------------------------------


def eudis2(v1, v2):
    return math.dist(v1, v2)
    # return distance.euclidean(v1, v2)


def calculate_spread(population, dataset):
    MIN_OBJ1 = 0
    MIN_OBJ2 = 0
    # MAX_OBJ1 = 25  # max_importancia_Stakeholder * max_prioridad_pbi_para_Stakeholder # TODO fix
    # MAX_OBJ2 = 40  # max estimacion de pbi

    MAX_OBJ1 = np.max(dataset.pbis_satisfaction_scaled)
    MAX_OBJ2 = np.max(dataset.pbis_cost_scaled)

    df = None
    dl = None
    davg = None
    sum_dist = None
    N = len(population)
    spread = None

    first_solution = population[0]
    last_solution = population[len(population) - 1]

    # obtener first_extreme=[score=0 (worst),cost=0 (best)] y last_extreme=[score=MAX_SCORE (best),cost=MAX_COST (worst)]
    first_extreme = [MIN_OBJ1, MIN_OBJ2]
    last_extreme = [MAX_OBJ1, MAX_OBJ2]

    df = eudis2([first_solution.total_satisfaction,
                first_solution.total_cost], first_extreme)
    dl = eudis2([last_solution.total_satisfaction,
                last_solution.total_cost], last_extreme)

    # calcular media de todas las distancias entre puntos
    davg = 0
    dist_count = 0
    for i in range(0, len(population)):
        for j in range(0, len(population)):
            # no calcular distancia de un punto a si mismo
            if i != j:
                dist_count += 1
                davg += eudis2([population[i].total_satisfaction, population[i].total_cost],
                               [population[j].total_satisfaction, population[j].total_cost])
    # media=distancia total / numero de distancias
    davg /= dist_count

    # calcular sumatorio(i=1->N-1) |di-davg|
    sum_dist = 0
    for i in range(0, len(population) - 1):
        di = eudis2([population[i].total_satisfaction, population[i].total_cost],
                    [population[i + 1].total_satisfaction, population[i + 1].total_cost])
        sum_dist += abs(di - davg)

    # formula spread
    spread = (df + dl + sum_dist) / (df + dl + (N - 1) * davg)
    return spread


def calculate_mean_bits_per_sol(solutions):
    genes = 0
    n_sols = len(solutions)
    for sol in solutions:
    	genes += np.count_nonzero(sol.selected)
    return genes/n_sols
