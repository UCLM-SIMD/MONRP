from algorithms.abstract_genetic.basegenetic_algorithm import BaseGeneticAlgorithm
from algorithms.genetic.genetic_executer import GeneticExecuter
from algorithms.genetic.genetic_utils import GeneticUtils

import copy
import time


class GeneticAlgorithm(BaseGeneticAlgorithm):
    def __init__(self, dataset_name="1", random_seed=None, population_length=20, max_generations=1000,
                 selection="tournament", selection_candidates=2,
                 crossover="onepoint", crossover_prob=0.9,
                 mutation="flipeachbit", mutation_prob=0.1,
                 replacement="elitism"):

        self.utils = GeneticUtils(
            random_seed, population_length, selection_candidates, crossover_prob, mutation_prob)
        self.executer = GeneticExecuter(algorithm=self)
        self.problem,self.dataset = self.utils.generate_dataset_problem(
            dataset_name=dataset_name)
        self.dataset_name = dataset_name

        self.population_length = population_length
        self.max_generations = max_generations
        self.random_seed = random_seed

        self.selection_scheme = selection
        self.selection_candidates = selection_candidates
        self.crossover_scheme = crossover
        self.crossover_prob = crossover_prob
        self.mutation_scheme = mutation
        self.mutation_prob = mutation_prob
        self.replacement_scheme = replacement

        self.best_individual = None
        self.population = None
        self.best_generation_avgValue = None
        self.best_generation = None

        self.evaluate = self.utils.evaluate
        self.calculate_last_generation_with_enhance = self.utils.calculate_last_generation_with_enhance
        self.generate_starting_population = self.utils.generate_starting_population

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

        self.file = str(self.__class__.__name__)+"-"+str(dataset_name)+"-"+str(random_seed)+"-"+str(population_length)+"-" +\
            str(max_generations)+"-"+selection+"-"+str(selection_candidates)+"-" +\
            str(crossover)+"-"+str(crossover_prob)+"-"+str(mutation) + \
            "-"+str(mutation_prob)+"-"+str(replacement)+".txt"

    def get_name(self):
        return "Genetic"

    # RUN ALGORITHM------------------------------------------------------------------

    def run(self):
        start = time.time()

        num_generations = 0
        returned_population = None
        self.best_generation_avgValue = 0
        self.best_generation = 0
        self.population = self.generate_starting_population()
        self.evaluate(self.population, self.best_individual)
        #print("Best individual score: ", self.best_individual.total_score)

        while num_generations < self.max_generations:
            # selection
            new_population = self.selection(self.population)

            # crossover
            new_population = self.crossover(new_population)

            # mutation
            new_population = self.mutation(new_population)

            # evaluation
            self.evaluate(self.population, self.best_individual)
            returned_population = copy.deepcopy(new_population)

            self.best_generation, self.best_generation_avgValue = self.calculate_last_generation_with_enhance(
                self.best_generation, self.best_generation_avgValue, num_generations, returned_population)

            # replacement
            self.population = self.replacement(self.population, new_population)

            num_generations += 1
            # mostrar por pantalla
            # if num_generations % 100 == 0:
            #print("Nº Generations: ", num_generations)
            #print("Best individual score: ", self.best_individual.total_score)

        # end
        # print(self.best_individual)

        end = time.time()

        return {"population": returned_population,
                "time": end - start,
                "numGenerations": num_generations,
                "bestGeneration": self.best_generation
                }
