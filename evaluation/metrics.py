import random
from models.population import Population
import math

# AVGVALUE------------------------------------------------------------------
def calculate_avgValue( population):
	avgValue = 0
	for ind in population:
		avgValue += ind.total_score
	avgValue /= len(population)
	return avgValue

# BESTAVGVALUE------------------------------------------------------------------
def calculate_bestAvgValue( population):
	bestAvgValue = 0
	for ind in population:
		if bestAvgValue < ind.total_score:
			bestAvgValue = ind.total_score

	return bestAvgValue

# NUMSOLUTIONS------------------------------------------------------------------
def calculate_numSolutions( population):
	return len(set(population))

# SPACING------------------------------------------------------------------
def calculate_spacing(population):
	n = len(population)
	N = len(population[0].objectives)
	spacing = 0
	mean_objectives = []
	# calcular la media de cada objetivo
	for i in range(0, len(population[0].objectives)):
		objective = 0
		for j in range(0, len(population)):
			objective += population[j].objectives[i].value
		objective /= len(population)
		mean_objectives.append(objective)

	for j in range(0, len(population)):
		aux_spacing = 0
		for i in range(0, len(population[0].objectives)):
			di = mean_objectives[i]
			dij = population[j].objectives[i].value
			aux = (1 - (abs(dij) / di)) ** 2
			aux_spacing += aux
		aux_spacing = math.sqrt(aux_spacing)
		spacing += aux_spacing

	spacing /= (n * N)
	return spacing

# HYPERVOLUME------------------------------------------------------------------
def calculate_hypervolume(population):
	# obtener minimos y maximos de cada objetivo
	objectives_diff=[]
	aux_max_obj=[population[0].max_score,population[0].max_cost]
	aux_min_obj=[population[0].min_score,population[0].min_cost]
	for i in range(0,len(population[0].objectives)):
		aux_min = float('inf')
		aux_max = 0
		for ind in population:  ##############################################
			if ind.objectives[i].value < aux_min:
				aux_min = ind.objectives[i].value
			if ind.objectives[i].value > aux_max:
				aux_max = ind.objectives[i].value

		aux_max_norm = (aux_max-aux_min_obj[i])/(aux_max_obj[i]-aux_min_obj[i])
		aux_min_norm = (aux_min-aux_min_obj[i])/(aux_max_obj[i]-aux_min_obj[i])
		aux_val = aux_max_norm-aux_min_norm
		objectives_diff.append(aux_val)

	# calcular hypervolume
	hypervolume=1
	for i in range(0, len(objectives_diff)):
		hypervolume*=objectives_diff[i]

	return hypervolume

# SPREAD------------------------------------------------------------------
def eudis2(v1, v2):
	return math.dist(v1, v2)
	#return distance.euclidean(v1, v2)

def calculate_spread(population):
	MIN_OBJ1 = 0
	MIN_OBJ2 = 0
	MAX_OBJ1 = 25  # max_importancia_Stakeholder * max_prioridad_pbi_para_Stakeholder # TODO fix 
	MAX_OBJ2 = 40  # max estimacion de pbi
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

	df = eudis2([first_solution.objectives[0].value, first_solution.objectives[1].value], first_extreme)
	dl = eudis2([last_solution.objectives[0].value, last_solution.objectives[1].value], last_extreme)

	# calcular media de todas las distancias entre puntos
	davg = 0
	dist_count = 0
	for i in range(0, len(population)):
		for j in range(0, len(population)):
			# no calcular distancia de un punto a si mismo
			if i != j:
				dist_count += 1
				davg += eudis2([population[i].objectives[0].value, population[i].objectives[1].value],
									[population[j].objectives[0].value, population[j].objectives[1].value])
	# media=distancia total / numero de distancias
	davg /= dist_count

	# calcular sumatorio(i=1->N-1) |di-davg|
	sum_dist = 0
	for i in range(0, len(population) - 1):
		di = eudis2([population[i].objectives[0].value, population[i].objectives[1].value],
						 [population[i + 1].objectives[0].value, population[i + 1].objectives[1].value])
		sum_dist += abs(di - davg)

	# formula spread
	spread = (df + dl + sum_dist) / (df + dl + (N - 1) * davg)
	return spread