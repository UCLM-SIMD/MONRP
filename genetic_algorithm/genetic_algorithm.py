from genetic_algorithm.genetic_utils import GeneticUtils
from individual import Individual
from population import Population
import copy


class GeneticAlgorithm:
	def __init__(self, problem, population_length=20, max_evaluations=1000,
				 selection="tournament", selection_candidates=2,
				 crossover="onepoint", crossover_prob=0.9,
				 mutation="mutation", mutation_prob=0.1,
				 replacement="elitism"):

		self.problem = problem
		self.population_length = population_length
		self.max_evaluations = max_evaluations

		self.utils = GeneticUtils(self.problem, selection_candidates, crossover_prob, mutation_prob)
		self.best_individual = None

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
		num_evaluations = 0
		population = self.generate_starting_population()
		self.evaluate(population)
		print("Best individual score: ", self.best_individual.total_score)
		while num_evaluations < self.max_evaluations:
			# selection
			new_population = self.selection(population)
			# crossover
			new_population = self.crossover(new_population)
			# mutation
			new_population = self.mutation(new_population)
			# evaluation
			self.evaluate(new_population)
			# replacement
			population = self.replacement(population, new_population)

			num_evaluations += 1
			# mostrar por pantalla
			if num_evaluations % 100 == 0:
				print("Nº Evaluations: ", num_evaluations)
				print("Best individual score: ", self.best_individual.total_score)

		# end
		# print(self.best_individual)
