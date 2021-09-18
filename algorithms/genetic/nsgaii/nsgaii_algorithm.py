from algorithms.genetic.abstract_genetic.basegenetic_algorithm import BaseGeneticAlgorithm
from algorithms.genetic.nsgaii.nsgaii_executer import NSGAIIExecuter
from algorithms.genetic.nsgaii.nsgaii_utils import NSGAIIUtils
from models.population import Population
import copy
import time


class NSGAIIAlgorithm(BaseGeneticAlgorithm):# TODO NSGAIIALGORITHM -> NSGAII y reescribir ficheros output
    def __init__(self, dataset_name="1", random_seed=None, population_length=20, max_generations=1000,max_evaluations=0,
                 selection="tournament", selection_candidates=2,
                 crossover="onepoint", crossover_prob=0.9,
                 mutation="flipeachbit", mutation_prob=0.1,
                 replacement="elitism"):

        self.utils = NSGAIIUtils(
            random_seed, population_length, selection_candidates, crossover_prob, mutation_prob)
        self.executer = NSGAIIExecuter(algorithm=self)
        self.problem, self.dataset = self.utils.generate_dataset_problem(
            dataset_name=dataset_name)
        self.dataset_name = dataset_name

        self.random_seed = random_seed
        self.population_length = population_length
        self.max_generations = max_generations
        self.max_evaluations = max_evaluations

        self.population = None
        self.best_generation_avgValue = None
        self.best_generation = None
        self.best_individual = None

        self.selection_scheme = selection
        self.selection_candidates = selection_candidates
        self.crossover_scheme = crossover
        self.crossover_prob = crossover_prob
        self.mutation_scheme = mutation
        self.mutation_prob = mutation_prob
        self.replacement_scheme = replacement

        self.fast_nondominated_sort = self.utils.fast_nondominated_sort
        self.calculate_crowding_distance = self.utils.calculate_crowding_distance
        self.crowding_operator = self.utils.crowding_operator
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
            str(max_generations) + "-"+str(max_evaluations)+ "-"+selection+"-"+str(selection_candidates)+"-" +\
            str(crossover)+"-"+str(crossover_prob)+"-"+str(mutation) + \
            "-"+str(mutation_prob)+"-"+str(replacement)+".txt"

    def get_name(self):
        return "NSGA-II+"+str(self.population_length)+"+"+str(self.max_generations)+"+"+str(self.max_evaluations)\
        +"+"+str(self.crossover_prob)\
            + "+"+str(self.mutation_scheme)+"+"+str(self.mutation_prob)

    def reset(self):
        self.best_generation_avgValue = 0
        self.best_generation = 0
        self.best_individual = None
        self.population = None

    # RUN ALGORITHM------------------------------------------------------------------
    def run(self):
        self.reset()
        start = time.time()

        num_evaluations = 0

        # inicializacion del nsgaii
        self.population = self.generate_starting_population()
        self.evaluate(self.population, self.best_individual)
        num_evaluations+=len(self.population)

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
        # or not(num_generations > (self.best_generation+20)):
       # while (num_generations < self.max_generations):
        while (self.stop_criterion(num_generations, num_evaluations)):
            self.population.extend(offsprings)
            self.evaluate(self.population, self.best_individual)
            num_evaluations+=len(self.population)

            self.fast_nondominated_sort(self.population)
            new_population = Population()
            front_num = 0

            # till parent population is filled, calculate crowding distance in Fi, include i-th non-dominated front in parent pop
            while len(new_population) + len(self.population.fronts[front_num]) <= self.population_length:
                self.calculate_crowding_distance(
                    self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1

            # ordenar los individuos del ultimo front por crowding distance y agregar los X que falten para completar la poblacion
            self.calculate_crowding_distance(self.population.fronts[front_num])

            # sort in descending order using >=n
            self.population.fronts[front_num].sort(
                key=lambda individual: individual.crowding_distance, reverse=True)

            # choose first N elements of Pt+1
            new_population.extend(
                self.population.fronts[front_num][0:self.population_length - len(new_population)])
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
            self.best_generation, self.best_generation_avgValue = self.calculate_last_generation_with_enhance(
                self.best_generation, self.best_generation_avgValue, num_generations, returned_population)

            num_generations += 1
            # mostrar por pantalla
            # if num_generations % 100 == 0:
            #	print("Nº Generations: ", num_generations)

        end = time.time()

        return {"population": returned_population.fronts[0],
                "time": end - start,
                "best_individual": self.best_individual,
                "bestGeneration": self.best_generation,
                "numGenerations": num_generations,
                "numEvaluations":num_evaluations
                }
