from algorithms.genetic.genetic_utils import GeneticUtils
from models.individual import Individual
from models.population import Population
import copy
import time

class GeneticAlgorithm:
	def __init__(self, problem, random_seed=None, population_length=20, max_generations=1000,
				 selection="tournament", selection_candidates=2,
				 crossover="onepoint", crossover_prob=0.9,
				 mutation="mutation", mutation_prob=0.1,
				 replacement="elitism"):

		self.problem = problem
		self.population_length = population_length
		self.max_generations = max_generations
		self.random_seed=random_seed

		self.selection_scheme = selection
		self.selection_candidates = selection_candidates
		self.crossover_scheme = crossover
		self.crossover_prob = crossover_prob
		self.mutation_scheme = mutation
		self.mutation_prob = mutation_prob
		self.replacement_scheme = replacement

		self.utils = GeneticUtils(self.problem,self.random_seed, selection_candidates, crossover_prob, mutation_prob)
		self.best_individual = None
		self.population=None
		self.best_generation_avgValue = None
		self.best_generation = None
		self.calculate_numSolutions = self.utils.calculate_numSolutions
		self.calculate_avgValue = self.utils.calculate_avgValue
		self.calculate_bestAvgValue = self.utils.calculate_bestAvgValue

		if selection == "tournament":
			self.selection = self.utils.selection_tournament

		if crossover == "onepoint":
			self.crossover = self.utils.crossover_one_point

		if mutation == "mutation":
			self.mutation = self.utils.mutation

		if replacement == "elitism":
			self.replacement = self.utils.replacement_elitism
		else:
			self.replacement = self.utils.replacement_elitism

	'''
	def aa(self):
		random.seed(self.random_seed)
		print(random.random())
		print(random.random())
	'''

	# GENERATE STARTING POPULATION------------------------------------------------------------------
	def generate_starting_population(self):
		population = Population()
		for i in range(0, self.population_length):
			individual = Individual(self.problem.genes, self.problem.objectives)
			individual.initRandom()
			population.append(individual)
		return population

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

	# LAST GENERATION ENHANCE------------------------------------------------------------------
	def calculate_last_generation_with_enhance(self, num_generation, population):
		bestAvgValue = self.calculate_bestAvgValue(population)
		if bestAvgValue > self.best_generation_avgValue:
			self.best_generation_avgValue = bestAvgValue
			self.best_generation = num_generation

	# RUN ALGORITHM------------------------------------------------------------------
	def run(self):
		start = time.time()

		num_generations = 0
		returned_population = None
		self.best_generation_avgValue = 0
		self.best_generation = 0
		self.population = self.generate_starting_population()
		self.evaluate(self.population)
		#print("Best individual score: ", self.best_individual.total_score)

		while num_generations < self.max_generations:
			# selection
			new_population = self.selection(self.population)

			# crossover
			new_population = self.crossover(new_population)

			# mutation
			new_population = self.mutation(new_population)

			# evaluation
			self.evaluate(new_population)
			returned_population = copy.deepcopy(new_population)

			self.calculate_last_generation_with_enhance(num_generations, returned_population)

			# replacement
			self.population = self.replacement(self.population, new_population)

			num_generations += 1
			# mostrar por pantalla
			#if num_generations % 100 == 0:
				#print("Nº Generations: ", num_generations)
				#print("Best individual score: ", self.best_individual.total_score)

		# end
		# print(self.best_individual)

		avgValue = self.calculate_avgValue(returned_population)
		bestAvgValue = self.calculate_bestAvgValue(returned_population)
		end = time.time()

		return {"population": returned_population,
				"time": end - start,
				"avgValue": avgValue,
				"bestAvgValue": bestAvgValue,
				"best_individual": self.best_individual,
				"best_generation_num": self.best_generation
				}
