from models.individual import Solution
from algorithms.nsgaii.nsgaii_utils import NSGAIIUtils
from models.population import Population
import copy
import time


class NSGAIIAlgorithm:
	def __init__(self, problem, random_seed=None, population_length=20, max_generations=1000,
				 selection="tournament", selection_candidates=2,
				 crossover="onepoint", crossover_prob=0.9,
				 mutation="flipeachbit", mutation_prob=0.1,
				 replacement="elitism"):

		self.problem = problem
		self.random_seed = random_seed
		self.population_length = population_length
		self.max_generations = max_generations
		self.population = None
		self.best_generation_avgValue = None
		self.best_generation = None

		self.selection_scheme = selection
		self.selection_candidates = selection_candidates
		self.crossover_scheme = crossover
		self.crossover_prob = crossover_prob
		self.mutation_scheme = mutation
		self.mutation_prob = mutation_prob
		self.replacement_scheme = replacement

		self.utils = NSGAIIUtils(self.problem, random_seed, selection_candidates, crossover_prob, mutation_prob)

		self.calculate_numSolutions = self.utils.calculate_numSolutions
		self.calculate_spacing = self.utils.calculate_spacing
		self.calculate_avgValue = self.utils.calculate_avgValue
		self.calculate_bestAvgValue = self.utils.calculate_bestAvgValue
		self.best_individual = None

		self.calculate_hypervolume = self.utils.calculate_hypervolume
		self.calculate_spread = self.utils.calculate_spread
		self.fast_nondominated_sort = self.utils.fast_nondominated_sort
		self.calculate_crowding_distance = self.utils.calculate_crowding_distance
		self.crowding_operator = self.utils.crowding_operator

		if selection == "tournament":
			self.selection = self.utils.selection_tournament

		if crossover == "onepoint":
			self.crossover = self.utils.crossover_one_point

		if mutation == "flip1bit":
			self.mutation = self.utils.mutation_flip1bit
		elif mutation == "flipeachbit":
			self.mutation = self.utils.mutation_flipeachbit

		if replacement == "elitism":
			self.replacement = self.utils.replacement_elitism
		else:
			self.replacement = self.utils.replacement_elitism

	# GENERATE STARTING POPULATION------------------------------------------------------------------
	def generate_starting_population(self):
		population = Population()
		for i in range(0, self.population_length):
			individual = Solution(self.problem.genes, self.problem.objectives)
			individual.initRandom()
			population.append(individual)
		return population

	# LAST GENERATION ENHANCE------------------------------------------------------------------
	def calculate_last_generation_with_enhance(self, num_generation, population):
		bestAvgValue = self.calculate_bestAvgValue(population)
		if bestAvgValue > self.best_generation_avgValue:
			self.best_generation_avgValue = bestAvgValue
			self.best_generation = num_generation

	# EVALUATION------------------------------------------------------------------
	def evaluate(self, population):
		best_score = 0
		new_best_individual = None
		for ind in population:
			ind.evaluate_fitness()
			if ind.total_score > best_score:
				new_best_individual = copy.deepcopy(ind)
				best_score = ind.total_score
		# print(best_score)
		# print(ind)

		if self.best_individual is not None:
			if new_best_individual.total_score > self.best_individual.total_score:
				self.best_individual = copy.deepcopy(new_best_individual)
		else:
			self.best_individual = copy.deepcopy(new_best_individual)

	# RUN ALGORITHM------------------------------------------------------------------
	def run(self):
		start = time.time()
		self.best_generation_avgValue = 0
		self.best_generation = 0
		# inicializacion del nsgaii
		self.population = self.generate_starting_population()
		self.evaluate(self.population)
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
		while (num_generations < self.max_generations) or not(num_generations>(self.best_generation+20) ):
			self.population.extend(offsprings)
			self.evaluate(self.population)
			self.fast_nondominated_sort(self.population)
			new_population = Population()
			front_num = 0

			# till parent population is filled, calculate crowding distance in Fi, include i-th non-dominated front in parent pop
			while len(new_population) + len(self.population.fronts[front_num]) <= self.population_length:
				self.calculate_crowding_distance(self.population.fronts[front_num])
				new_population.extend(self.population.fronts[front_num])
				front_num += 1

			# ordenar los individuos del ultimo front por crowding distance y agregar los X que falten para completar la poblacion
			self.calculate_crowding_distance(self.population.fronts[front_num])

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
			self.calculate_last_generation_with_enhance(num_generations, returned_population)

			num_generations += 1
			# mostrar por pantalla
			#if num_generations % 100 == 0:
			#	print("Nº Generations: ", num_generations)

		end = time.time()

		return {"population": returned_population.fronts[0],
				"time": end - start,
				"best_individual": self.best_individual,
				"bestGeneration": self.best_generation,
				"numGenerations": num_generations,
				}
