from genetic_algorithm.genetic_utils import GeneticUtils
from individual import Individual
from nsgaii.nsgaii_utils import NSGAIIUtils
from population import Population
import copy
import time


class NSGAIIAlgorithm:
	def __init__(self, problem, random_seed, population_length=20, max_generations=1000,
				 selection="tournament", selection_candidates=2,
				 crossover="onepoint", crossover_prob=0.9,
				 mutation="mutation", mutation_prob=0.1,
				 replacement="elitism"):

		self.problem = problem
		self.random_seed = random_seed
		self.population_length = population_length
		self.max_generations = max_generations

		self.population = None

		self.utils = NSGAIIUtils(self.problem, random_seed, selection_candidates, crossover_prob, mutation_prob)

		self.calculate_hypervolume = self.utils.calculate_hypervolume
		self.calculate_spread = self.utils.calculate_spread
		self.fast_nondominated_sort = self.utils.fast_nondominated_sort
		self.calculate_crowding_distance = self.utils.calculate_crowding_distance
		self.crowding_operator = self.utils.crowding_operator

		if selection == "tournament":
			self.selection = self.utils.selection_tournament

		if crossover == "onepoint":
			self.crossover = self.utils.crossover_one_point

		if mutation == "mutation":
			self.mutation = self.utils.mutation

		if replacement == "elitism":
			self.replacement = self.utils.replacement_elitism

	# GENERATE STARTING POPULATION------------------------------------------------------------------
	def generate_starting_population(self):
		population = Population()
		for i in range(0, self.population_length):
			individual = Individual(self.problem.genes, self.problem.objectives)
			individual.initRandom()
			population.append(individual)
		return population

	# RUN ALGORITHM------------------------------------------------------------------
	def run(self):
		start = time.time()
		# inicializacion del nsgaii
		self.population = self.generate_starting_population()

		# ordenar por NDS y crowding distance
		self.fast_nondominated_sort(self.population)
		for front in self.population.fronts:
			self.calculate_crowding_distance(front)

		# crear hijos
		offsprings = self.selection(self.population)
		offsprings = self.crossover(offsprings)
		offsprings = self.mutation(offsprings)
		# offsprings = self.replacement(self.population, offsprings)

		# iteraciones del nsgaii
		num_generations = 0
		returned_population = None
		while num_generations < self.max_generations:
			self.population.extend(offsprings)
			self.fast_nondominated_sort(self.population)
			new_population = Population()
			front_num = 0

			# till parent population is filled, calculate crowding distance in Fi, include i-th non-dominated front in parent pop
			while len(new_population) + len(self.population.fronts[front_num]) <= self.population_length:
				self.calculate_crowding_distance(self.population.fronts[front_num])
				new_population.extend(self.population.fronts[front_num])
				front_num += 1

			# ordenar los individuos del ultimo front por crowding distance y agregar los X que falten para completar la poblacion
			self.calculate_crowding_distance(self.population.fronts[front_num])  ###########no se por que

			# sort in descending order using >=n
			self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)

			# choose first N elements of Pt+1
			new_population.extend(self.population.fronts[front_num][0:self.population_length - len(new_population)])
			self.population = copy.deepcopy(new_population)

			# ordenar por NDS y crowding distance
			self.fast_nondominated_sort(self.population)
			for front in self.population.fronts:
				self.calculate_crowding_distance(front)

			# use selection,crossover and mutation to create a new population Qt+1
			offsprings = self.selection(self.population)
			offsprings = self.crossover(offsprings)
			offsprings = self.mutation(offsprings)
			# offsprings = self.replacement(self.population, offsprings)

			returned_population = copy.deepcopy(self.population)
			num_generations += 1
			# mostrar por pantalla
			if num_generations % 100 == 0:
				print("Nº Generations: ", num_generations)

		end = time.time()

		hv = self.calculate_hypervolume(returned_population)
		spread = self.calculate_spread(returned_population)
		return {"population": returned_population.fronts[0],
				"time": end - start,
				"hv": hv,
				"spread": spread
				}
