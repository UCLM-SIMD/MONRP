import numpy as np
from algorithms.GRASP.GraspSolution import GraspSolution
from evaluation.format_population import format_population
from algorithms.abstract_default.evaluation_exception import EvaluationLimit
from algorithms.genetic.abstract_genetic.basegenetic_algorithm import BaseGeneticAlgorithm
from algorithms.genetic.geneticnds.geneticnds_executer import GeneticNDSExecuter
from algorithms.genetic.geneticnds.geneticnds_utils import GeneticNDSUtils
import copy
import time
from evaluation.update_nds import get_nondominated_solutions
from models.population import Population
import random
import evaluation.metrics as metrics

class GeneticNDSAlgorithm(BaseGeneticAlgorithm):
    def __init__(self, dataset_name: str = "test", random_seed: int = None, debug_mode: bool = False, tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0,
                 selection: str = "tournament", selection_candidates: int = 2,
                 crossover: str = "onepoint", crossover_prob: float = 0.9,
                 mutation: str = "flipeachbit", mutation_prob: float = 0.1,
                 replacement: str = "elitism",):

        super().__init__(dataset_name, random_seed, debug_mode, tackle_dependencies,
                         population_length, max_generations, max_evaluations,
                         selection, selection_candidates, crossover, crossover_prob,
                         mutation, mutation_prob, replacement,)

        #self.utils = GeneticNDSUtils(
        #    random_seed, population_length, selection_candidates, crossover_prob, mutation_prob)
        self.executer = GeneticNDSExecuter(algorithm=self)
        #self.problem, self.dataset = self.utils.generate_dataset_problem(
        #    dataset_name=dataset_name)
        self.dataset_name = dataset_name

        #self.population_length = population_length
        #self.max_generations = max_generations
        #self.max_evaluations = max_evaluations
        #self.random_seed = random_seed

        #self.selection_scheme = selection
        #self.selection_candidates = selection_candidates
        #self.crossover_scheme = crossover
        #self.crossover_prob = crossover_prob
        #self.mutation_scheme = mutation
        #self.mutation_prob = mutation_prob
        #self.replacement_scheme = replacement

        #self.population = None
        #self.best_generation_avgValue = None
        #self.best_generation = None

        #self.nds = []
        #self.num_evaluations: int = 0
        #self.num_generations: int = 0
        #self.best_individual = None

        #self.debug_mode = debug_mode
        #self.tackle_dependencies = tackle_dependencies

        #self.evaluate = self.utils.evaluate

        #self.calculate_last_generation_with_enhance = self.calculate_last_generation_with_enhance
        #self.generate_starting_population = self.generate_starting_population
        #self.repair_population_dependencies = self.repair_population_dependencies

        if selection == "tournament":
            self.selection = self.selection_tournament

        if crossover == "onepoint":
            self.crossover = self.crossover_one_point

        if mutation == "flip1bit":
            self.mutation = self.mutation_flip1bit
        elif mutation == "flipeachbit":
            self.mutation = self.mutation_flipeachbit

        if replacement == "elitism":
            self.replacement = self.replacement_elitism
        elif replacement == "elitismnds":
            self.replacement = self.replacement_elitism

        self.file: str = str(self.__class__.__name__)+"-"+str(dataset_name)+"-"+str(random_seed)+"-"+str(population_length)+"-" +\
            str(max_generations)+"-" +\
            selection+"-"+str(selection_candidates)+"-" +\
            str(crossover)+"-"+str(crossover_prob)+"-"+str(mutation) + \
            "-"+str(mutation_prob)+"-"+str(replacement)+".txt"
        # + "-"+str(max_evaluations) TODO

    def get_name(self):
        return "GeneticNDS+"+str(self.population_length)+"+"+str(self.max_generations) + \
            "+"+str(self.max_evaluations)+"+"+str(self.crossover_prob)\
            + "+"+str(self.mutation_scheme)+"+"+str(self.mutation_prob)

    # UPDATE NDS------------------------------------------------------------------

    #def is_non_dominated(self, ind, nds):
    #    non_dominated = True
    #    for other_ind in nds:
    #        # if ind.dominates(other_ind):
    #        # non_dominated=non_dominated and True
    #        #	pass
    #        # elif other_ind.dominates(ind):
    #        if other_ind.dominates(ind):
    #            non_dominated = False
    #            break
#
    #    return non_dominated
#
    #def updateNDS(self, new_population):
    #    new_nds = []
    #    merged_population = copy.deepcopy(self.nds)
    #    merged_population.extend(new_population)
    #    for ind in merged_population:
    #        dominated = False
    #        for other_ind in merged_population:
    #            if other_ind.dominates(ind):
    #                dominated = True
    #                break
    #        if not dominated:
    #            new_nds.append(ind)
    #    new_nds = list(set(new_nds))
    #    self.nds = copy.deepcopy(new_nds)

    # def evaluate(self, population, best_individual):
    #    #super().evaluate(population, best_individual)
    #    try:
    #        best_score = 0
    #        new_best_individual = None
    #        for ind in population:
    #            ind.evaluate_fitness()
    #            self.add_evaluation(population)
    #            if ind.total_score > best_score:
    #                new_best_individual = copy.deepcopy(ind)
    #                best_score = ind.total_score
    #        if best_individual is not None:
    #            if new_best_individual.total_score > best_individual.total_score:
    #                best_individual = copy.deepcopy(new_best_individual)
    #        else:
    #            best_individual = copy.deepcopy(new_best_individual)
    #    except EvaluationLimit:
    #        pass

    # def add_evaluation(self, new_population):
    #    self.num_evaluations += 1
    #    # if(self.num_evaluations >= self.max_evaluations):
    #    if (self.stop_criterion(self.num_generations, self.num_evaluations)):
    #        self.updateNDS(new_population)
    #        raise EvaluationLimit

    def reset(self):
        super().reset()
        self.nds = []

    # RUN ALGORITHM------------------------------------------------------------------

    def run(self):
        self.reset()
        paretos = []
        start = time.time()

        self.num_generations = 0
        self.num_evaluations = 0
        self.population = self.generate_starting_population()
        self.evaluate(self.population, self.best_individual)
        # print("Best individual score: ", self.best_individual.total_score)

        # or not(num_generations > (self.best_generation+20)):
        # while (num_generations < self.max_generations):
        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                # selection
                new_population = self.selection(self.population)
                # crossover
                new_population = self.crossover(new_population)

                # mutation
                new_population = self.mutation(new_population)

                # repair population if dependencies tackled:
                if(self.tackle_dependencies):
                    new_population = self.repair_population_dependencies(
                        new_population)

                # evaluation
                self.evaluate(self.population, self.best_individual)
                # num_evaluations+=len(self.population)

                # update NDS
                # self.updateNDS(new_population)
                get_nondominated_solutions(new_population, self.nds)

                returned_population = copy.deepcopy(new_population)
                self.best_generation, self.best_generation_avgValue = self.calculate_last_generation_with_enhance(
                    self.best_generation, self.best_generation_avgValue, self.num_generations, returned_population)

                # replacement
                if self.replacement_scheme == "elitismnds":
                    self.population = self.replacement(
                        self.nds, new_population)
                else:
                    self.population = self.replacement(
                        self.population, new_population)

                self.num_generations += 1
                if self.debug_mode:
                    paretos.append(self.nds)

                # mostrar por pantalla
                # if num_generations % 100 == 0:
                # print("Nº Generations: ", num_generations)
                # print("Best individual score: ", self.best_individual.total_score)

        except EvaluationLimit:
            pass

        # end
        # print(self.best_individual)
        end = time.time()

        return {  # "population": returned_population,
            "population": self.nds,
            "time": end - start,
            "best_individual": self.best_individual,
            # "nds": self.nds,
            "bestGeneration": self.best_generation,
            "numGenerations": self.num_generations,
            "numEvaluations": self.num_evaluations,
            "paretos": paretos
        }

    def calculate_last_generation_with_enhance(self, best_generation, best_generation_avgValue, num_generation, population):
        bestAvgValue = metrics.calculate_bestAvgValue(population)
        if bestAvgValue > best_generation_avgValue:
            best_generation_avgValue = bestAvgValue
            best_generation = num_generation
        return best_generation, best_generation_avgValue


    def add_evaluation(self, new_population):
        self.num_evaluations += 1
        # if(self.num_evaluations >= self.max_evaluations):
        if (self.stop_criterion(self.num_generations, self.num_evaluations)):
            self.update_nds(new_population)
            raise EvaluationLimit

    def selection_tournament(self, population):
        new_population = []  # Population()
        # crear tantos individuos como tamaño de la poblacion
        for i in range(0, len(population)):
            best_candidate = None
            best_total_score = 0
            # elegir individuo entre X num de candidatos aleatorios
            for j in range(0, self.selection_candidates):
                random_index = random.randint(0, len(population)-1)
                #candidate = population.get(random_index)
                candidate = population[random_index]
                # candidate.evaluate_fitness()

                score = candidate.total_satisfaction
                cost = candidate.total_cost
                total_score = candidate.mono_objective_score
                # guardar el candidato con mejor score
                if(total_score > best_total_score):
                    best_total_score = total_score
                    best_candidate = candidate

            # insertar el mejor candidato del torneo en la nueva poblacion
            new_population.append(best_candidate)

        # retornar nueva poblacion
        return new_population

    # CROSSOVER------------------------------------------------------------------

    #def crossover_one_point(self, population):
    #    new_population = []  # Population()
    #    i = 0
    #    while i < len(population):
    #        # if last element is alone-> add it
    #        if i == len(population) - 1:
    #            new_population.append(population[i])  # .get(i))
#
    #        else:
    #            # pair 2 parents -> crossover or add them and jump 1 index extra
    #            prob = random.random()
    #            if prob < self.crossover_prob:
    #                offsprings = self.crossover_aux_one_point(
    #                    population[i], population[i+1])
    #                # print(offsprings)
    #                new_population.extend(offsprings)
    #            else:
    #                new_population.extend(
    #                    [population[i], population[i+1]])
    #            i += 1
#
    #        i += 1
#
    #    return new_population

#    def crossover_aux_one_point(self, parent1, parent2):
#        chromosome_length = len(parent1.selected)
#        # index aleatorio del punto de division para el cruce
#        crossover_point = random.randint(1, chromosome_length - 1)
#
#        offspring_genes1 = np.concatenate((parent1.selected[0:crossover_point],
#        parent2.selected[crossover_point:]) )
#        #parent1.selected[0:crossover_point] + \
#        #    parent2.selected[crossover_point:]
#
#
#        offspring_genes2 = np.concatenate((parent2.selected[0:crossover_point],
#        parent1.selected[crossover_point:]) )
#        # parent2.selected[0:crossover_point] + \
#        #    parent1.selected[crossover_point:]
#
#        # self.problem.generate_individual(
#        offspring1 = GraspSolution(
#            self.dataset, None, selected=offspring_genes1)
#        # offspring_genes1, self.dataset.dependencies)
#        # self.problem.generate_individual(
#        offspring2 = GraspSolution(
#            self.dataset, None, selected=offspring_genes2)
#        # offspring_genes2, self.dataset.dependencies)
#
#        return offspring1, offspring2

    # MUTATION------------------------------------------------------------------

#    def mutation_flip1bit(self, population):
#        new_population = []  # Population()
#        new_population.extend(population)  # .population)
#
#        for individual in new_population:
#            prob = random.random()
#            if prob < self.mutation_prob:
#                chromosome_length = len(individual.selected)
#                mutation_point = random.randint(0, chromosome_length - 1)
#                if individual.selected[mutation_point] == 0:
#                    individual.set_bit(mutation_point, 1)
#                else:
#                    individual.set_bit(mutation_point, 0)
#
#        return new_population
#
#    def mutation_flipeachbit(self, population):
#        new_population = []  # Population()
#        new_population.extend(population)  # .population)
#
#        for individual in new_population:
#            for gen_index in range(len(individual.selected)):
#                prob = random.random()
#                if prob < self.mutation_prob:
#                    if individual.selected[gen_index] == 0:
#                        individual.set_bit(gen_index, 1)
#                    else:
#                        individual.set_bit(gen_index, 0)
#
#        return new_population

    # REPLACEMENT------------------------------------------------------------------
    def replacement_elitism(self, population, newpopulation):
        # encontrar mejor individuo de poblacion
        best_individual = None
        best_individual_total_score = 0
        for ind in population:
            if (ind.mono_objective_score > best_individual_total_score):
                best_individual_total_score = ind.mono_objective_score
                best_individual = ind

        # encontrar indice del peor individuo de nueva poblacion
        newpopulation_replaced = []
        newpopulation_replaced.extend(newpopulation)

        worst_individual_total_score = float('inf')
        worst_individual_index = None
        for ind in newpopulation_replaced:
            if (ind.mono_objective_score < worst_individual_total_score):
                worst_individual_total_score = ind.mono_objective_score
                worst_individual_index = newpopulation_replaced.index(ind)

        # reemplazar peor individuo por el mejor de poblacion antigua
        newpopulation_replaced[worst_individual_index]=best_individual
        return newpopulation_replaced
