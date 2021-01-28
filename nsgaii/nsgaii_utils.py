import math
from random import randint, random
import copy

from population import Population


class NSGAIIUtils:
	def __init__(self, problem, selection_candidates=2, crossover_prob=0.9, mutation_prob=0.1):
		self.problem = problem
		self.selection_candidates = selection_candidates
		self.crossover_prob = crossover_prob
		self.mutation_prob = mutation_prob

	# SELECTION------------------------------------------------------------------
	def selection_tournament(self, population):
		self.num_candidates = 2
		new_population = Population()
		# crear tantos individuos como tamaño de la poblacion
		for i in range(0, len(population)):
			best_candidate = None
			# elegir individuo entre X num de candidatos aleatorios
			for j in range(0, self.num_candidates):
				random_index = randint(0, len(population) - 1)
				candidate = population.get(random_index)

				# guardar el candidato que mejor crowding tiene
				if (best_candidate is None or self.crowding_operator(candidate,best_candidate)==1):
					best_candidate = copy.deepcopy(candidate)

			# insertar el mejor candidato del torneo en la nueva poblacion
			new_population.append(best_candidate)

		# retornar nueva poblacion
		return new_population

	# CROSSOVER------------------------------------------------------------------
	def crossover_one_point(self, population):
		new_population = Population()

		# for i in range(0, len(population)-1):
		i = 0
		while i < len(population):
			# if last element is alone-> add it
			if i == len(population) - 1:
				new_population.append(population.get(i))

			else:
				# pair 2 parents -> crossover or add them and jump 1 index extra
				prob = random()
				if prob < self.crossover_prob:
					offsprings = self.crossover_aux_one_point(population.get(i), population.get(i + 1))
					new_population.extend(offsprings)
				else:
					new_population.extend([population.get(i), population.get(i + 1)])
				i += 1

			i += 1

		return new_population

	def crossover_aux_one_point(self, parent1, parent2):
		chromosome_length = len(parent1.genes)

		# index aleatorio del punto de division para el cruce
		crossover_point = randint(1, chromosome_length - 1)
		offspring_genes1 = parent1.genes[0:crossover_point] + parent2.genes[crossover_point:]
		offspring_genes2 = parent2.genes[0:crossover_point] + parent1.genes[crossover_point:]

		offspring1 = self.problem.generate_individual(offspring_genes1)
		offspring2 = self.problem.generate_individual(offspring_genes2)

		return offspring1, offspring2

	# MUTATION------------------------------------------------------------------
	def mutation(self, population):
		new_population = Population()
		new_population.extend(population.population)

		for individual in new_population:
			prob = random()
		if prob < self.mutation_prob:
			chromosome_length = len(individual.genes)
			mutation_point = randint(0, chromosome_length - 1)
			if individual.genes[mutation_point].included == 0:
				individual.genes[mutation_point].included = 1
			else:
				individual.genes[mutation_point].included = 0

		return new_population

	# REPLACEMENT------------------------------------------------------------------
	def replacement_elitism(self, population, newpopulation):
		# encontrar mejor individuo de poblacion
		best_individual = None
		for ind in population:
			if (best_individual is None or self.crowding_operator(ind, best_individual) == 1):
				best_individual = copy.deepcopy(ind)

		newpopulation_replaced = Population()
		newpopulation_replaced.extend(newpopulation.population)
		worst_individual_index = None
		worst_individual = None
		for ind in newpopulation_replaced:
			if (worst_individual is None or self.crowding_operator(ind, worst_individual) == -1):
				worst_individual = copy.deepcopy(ind)
				worst_individual_index = newpopulation_replaced.index(ind)

		# reemplazar peor individuo por el mejor de poblacion antigua
		newpopulation_replaced.set(worst_individual_index, best_individual)
		return newpopulation_replaced

	# FAST NONDOMINATED SORT------------------------------------------------------------------
	def fast_nondominated_sort(self, population):
		population.fronts = [[]]
		for individual in population:
			individual.domination_count = 0
			individual.dominated_solutions = []
			for other_individual in population:
				if individual.dominates(other_individual):
					individual.dominated_solutions.append(other_individual)
				elif other_individual.dominates(individual):
					individual.domination_count += 1
			if individual.domination_count == 0:
				individual.rank = 0
				population.fronts[0].append(individual)
		i = 0
		while len(population.fronts[i]) > 0:
			temp = []
			for individual in population.fronts[i]:
				for other_individual in individual.dominated_solutions:
					other_individual.domination_count -= 1
					if other_individual.domination_count == 0:
						other_individual.rank = i + 1
						temp.append(other_individual)
			i = i + 1
			population.fronts.append(temp)

	# CALCULATE CROWDING DISTANCE------------------------------------------------------------------
	def calculate_crowding_distance(self, front):
		if len(front) > 0:
			solutions_num = len(front)
			for individual in front:
				individual.crowding_distance = 0

			for m in range(len(front[0].objectives)):
				front.sort(key=lambda individual: individual.objectives[m].value)
				# front[0].crowding_distance = 10 ** 9 #########################
				front[0].crowding_distance = float('inf')
				# front[solutions_num - 1].crowding_distance = 10 ** 9 #########################
				front[solutions_num - 1].crowding_distance = float('inf')
				m_values = [individual.objectives[m].value for individual in front]
				scale = max(m_values) - min(
					m_values)  #########################aqui calcula la escala o diferencia entre el valor mayor y el menor, y la usa para dividir en el crowding distance
				if scale == 0: scale = 1
				for i in range(1, solutions_num - 1):
					front[i].crowding_distance += (front[i + 1].objectives[m].value - front[i - 1].objectives[m].value) / scale

	# CROWDING OPERATOR------------------------------------------------------------------
	def crowding_operator(self, individual, other_individual):
		if (individual.rank < other_individual.rank) or \
				((individual.rank == other_individual.rank) and (
						individual.crowding_distance > other_individual.crowding_distance)):
			return 1
		else:
			return -1

	# HYPERVOLUME------------------------------------------------------------------
	def calculate_hypervolume(self, population):
		# obtener minimos y maximos de cada objetivo
		score_min = float('inf')
		score_max = 0
		cost_min = float('inf')
		cost_max = 0
		for ind in population:
			if ind.score > score_max:
				score_max = ind.score
			if ind.score < score_min:
				score_min = ind.score

			if ind.cost > cost_max:
				cost_max = ind.cost
			if ind.cost < cost_min:
				cost_min = ind.cost

		# calcular hypervolume
		score_diff = score_max - score_min
		cost_diff = cost_max - cost_min
		hypervolume = score_diff * cost_diff
		return hypervolume

	# SPREAD------------------------------------------------------------------
	def eudis(self, v1, v2):
		dist = [(a - b) ** 2 for a, b in zip(v1, v2)]
		dist = math.sqrt(sum(dist))
		return dist

	def calculate_spread(self, population):
		MAX_SCORE = 25  # max_importancia_Stakeholder * max_prioridad_pbi_para_Stakeholder
		MAX_COST = 40  # max estimacion de pbi
		df = None
		dl = None
		davg = None
		sum_dist = None
		N = len(population)
		spread = None
		# ordenar de menor a mayor coste
		population.population.sort(key=lambda x: x.cost)
		# for p in population:
		# print(p)
		# obtener first_solution=menor coste y last_solution=mayor coste
		first_solution = population.get(0)
		last_solution = population.get(len(population) - 1)

		# obtener first_extreme=[score=0 (worst),cost=0 (best)] y last_extreme=[score=MAX_SCORE (best),cost=MAX_COST (worst)]
		first_extreme = [0, 0]
		last_extreme = [MAX_SCORE, MAX_COST]

		df = self.eudis([first_solution.score, first_solution.cost], first_extreme)
		dl = self.eudis([last_solution.score, last_solution.cost], last_extreme)

		# calcular media de todas las distancias entre puntos
		davg = 0
		dist_count = 0
		for i in range(0, len(population)):
			for j in range(0, len(population)):
				# no calcular distancia de un punto a si mismo
				if i != j:
					dist_count += 1
					davg += self.eudis([population.get(i).score, population.get(i).cost],
									   [population.get(j).score, population.get(j).cost])
		# media=distancia total / numero de distancias
		davg /= dist_count

		# calcular sumatorio(i=1->N-1) |di-davg|
		sum_dist = 0
		for i in range(0, len(population) - 1):
			di = self.eudis([population.get(i).score, population.get(i).cost],
							[population.get(i + 1).score, population.get(i + 1).cost])
			sum_dist += abs(di - davg)

		# formula spread
		spread = (df + dl + sum_dist) / (df + dl + (N - 1) * davg)
		return spread
