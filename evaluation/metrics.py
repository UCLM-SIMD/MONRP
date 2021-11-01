import math
from typing import List
import numpy as np
from datasets.Dataset import Dataset

from models.Solution import Solution


def calculate_avgValue(population: List[Solution]) -> float:
    avgValue = 0
    for ind in population:
        avgValue += ind.compute_mono_objective_score()
    avgValue /= len(population)
    return avgValue


def calculate_bestAvgValue(population: List[Solution]) -> float:
    bestAvgValue = 0
    for ind in population:
        if bestAvgValue < ind.compute_mono_objective_score():
            bestAvgValue = ind.compute_mono_objective_score()

    return bestAvgValue


def calculate_numSolutions(population: List[Solution]) -> int:
    return len(set(population))


def calculate_spacing(population: List[Solution]) -> float:
    n = len(population)
    N = 2
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

    for j in range(0, len(population)):
        aux_spacing = 0
        for i in range(0, N):
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


def calculate_hypervolume(population: List[Solution]) -> float:
    objectives_diff = []
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

    hypervolume = 1
    for i in range(0, len(objectives_diff)):
        hypervolume *= objectives_diff[i]

    return hypervolume


def eudis2(v1: float, v2: float) -> float:
    return math.dist(v1, v2)
    # return distance.euclidean(v1, v2)


def calculate_spread(population: List[Solution], dataset: Dataset) -> float:
    MIN_OBJ1 = 0
    MIN_OBJ2 = 0

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

    first_extreme = [MIN_OBJ1, MIN_OBJ2]
    last_extreme = [MAX_OBJ1, MAX_OBJ2]

    df = eudis2([first_solution.total_satisfaction,
                first_solution.total_cost], first_extreme)
    dl = eudis2([last_solution.total_satisfaction,
                last_solution.total_cost], last_extreme)

    davg = 0
    dist_count = 0
    for i in range(0, len(population)):
        for j in range(0, len(population)):
            # avoid distance from a point to itself
            if i != j:
                dist_count += 1
                davg += eudis2([population[i].total_satisfaction, population[i].total_cost],
                               [population[j].total_satisfaction, population[j].total_cost])
    davg /= dist_count

    # calculate sumatory(i=1->N-1) |di-davg|
    sum_dist = 0
    for i in range(0, len(population) - 1):
        di = eudis2([population[i].total_satisfaction, population[i].total_cost],
                    [population[i + 1].total_satisfaction, population[i + 1].total_cost])
        sum_dist += abs(di - davg)

    # spread formula
    spread = (df + dl + sum_dist) / (df + dl + (N - 1) * davg)
    return spread


def calculate_mean_bits_per_sol(solutions: List[Solution]) -> float:
    genes = 0
    n_sols = len(solutions)
    for sol in solutions:
        genes += np.count_nonzero(sol.selected)
    return genes/n_sols
