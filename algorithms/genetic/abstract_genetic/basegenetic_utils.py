from models.solution import Solution
from algorithms.abstract_default.utils import Utils
import random
from models.population import Population
#from scipy.spatial import distance
import math
import evaluation.metrics as metrics
import copy
from models.problem import Problem


class BaseGeneticUtils(Utils):
    def __init__(self, random_seed, population_length=20, selection_candidates=2, crossover_prob=0.9, mutation_prob=0.1):
        self.problem = None
        self.random_seed = random_seed
        random.seed(self.random_seed)
        self.selection_candidates = selection_candidates
        self.population_length = population_length
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.objectives_minimization = ["MAX", "MIN"]

    # GENERATE DATASET PROBLEM------------------------------------------------------------------
    #def generate_dataset_problem(self, dataset_name):
    #    genes, dataset = generate_dataset_genes(dataset_name)
    #    problem = Problem(genes, self.objectives_minimization)
    #    self.problem = problem
    #    self.dataset = dataset
    #    return self.problem, self.dataset

    # EVALUATION------------------------------------------------------------------
    def evaluate(self, population, best_individual):
        best_score = 0
        new_best_individual = None
        for ind in population:
            ind.evaluate_fitness()
            if ind.total_score > best_score:
                new_best_individual = copy.deepcopy(ind)
                best_score = ind.total_score
                # print(best_score)
            # print(ind)

        if best_individual is not None:
            if new_best_individual.total_score > best_individual.total_score:
                best_individual = copy.deepcopy(new_best_individual)
        else:
            best_individual = copy.deepcopy(new_best_individual)

    # GENERATE STARTING POPULATION------------------------------------------------------------------
    def generate_starting_population(self):
        population = Population()
        for i in range(0, self.population_length):
            individual = Solution(self.problem.genes,
                                  self.problem.objectives, self.dataset.dependencies)
            individual.initRandom()
            population.append(individual)
        return population

    # LAST GENERATION ENHANCE------------------------------------------------------------------
    def calculate_last_generation_with_enhance(self, best_generation, best_generation_avgValue, num_generation, population):
        bestAvgValue = metrics.calculate_bestAvgValue(population)
        if bestAvgValue > best_generation_avgValue:
            best_generation_avgValue = bestAvgValue
            best_generation = num_generation
        return best_generation, best_generation_avgValue

    def repair_population_dependencies(self, population):
        for ind in population:
            ind.correct_dependencies()
        return population
