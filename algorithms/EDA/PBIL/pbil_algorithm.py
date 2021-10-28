from algorithms.EDA.eda_algorithm import EDAAlgorithm
from evaluation.format_population import format_population
from evaluation.update_nds import get_nondominated_solutions
from algorithms.abstract_default.evaluation_exception import EvaluationLimit
from models.solution import Solution
from datasets.dataset_gen_generator import generate_dataset_genes
from models.problem import Problem
from algorithms.EDA.PBIL.pbil_executer import PBILExecuter
from algorithms.GRASP.GraspSolution import GraspSolution
from algorithms.GRASP.Dataset import Dataset
from algorithms.genetic.abstract_genetic.basegenetic_algorithm import BaseGeneticAlgorithm
from algorithms.genetic.genetic.genetic_executer import GeneticExecuter
from algorithms.genetic.genetic.genetic_utils import GeneticUtils

import copy
import time
import numpy as np


class PBILAlgorithm(EDAAlgorithm):  # Population Based Incremental Learning
    def __init__(self, dataset_name:str="1", random_seed:int=None, debug_mode:bool=False, tackle_dependencies:bool=False,
                population_length:int=100, max_generations:int=100, max_evaluations:int=0,
                 learning_rate:float=0.1, mutation_prob:float=0.1, mutation_shift:float=0.1):

        
        super().__init__(dataset_name, random_seed, debug_mode, tackle_dependencies,
            population_length, max_generations, max_evaluations)

        self.executer = PBILExecuter(algorithm=self)
        # self.problem, self.dataset = self.utils.generate_dataset_problem(
        #    dataset_name=dataset_name)

        #self.dataset = Dataset(dataset_name)
        #self.dataset_name = dataset_name

        #self.population_length = population_length
        #self.max_generations = max_generations
        #self.max_evaluations = max_evaluations

        self.learning_rate:float = learning_rate
        self.mutation_prob:float = mutation_prob
        self.mutation_shift:float = mutation_shift

        #self.nds = []
        #self.best_individual = None
        #self.num_generations = 0
        #self.num_evaluations = 0

        #self.debug_mode = debug_mode
        #self.tackle_dependencies = tackle_dependencies

        #self.random_seed = random_seed
        #if random_seed is not None:
        #    np.random.seed(random_seed)

        self.file:str = str(self.__class__.__name__)+"-"+str(dataset_name)+"-"+str(random_seed)+"-"+str(population_length)+"-" +\
            str(max_generations) + "-"+str(max_evaluations)+"-"+str(learning_rate)+"-" + \
            str(mutation_prob)+"-"+str(mutation_shift)+".txt"

    def get_name(self):
        return f"PBIL+{self.population_length}+{self.max_generations}+{self.max_evaluations}+{self.learning_rate}+{self.mutation_prob}+{self.mutation_shift}"

    def initialize_probability_vector(self):
        probabilities = np.full(self.dataset.pbis_score.size, 0.5)
        #probabilities = np.full(self.dataset.pbis_score.size, 1/self.dataset.pbis_score.size)

        return probabilities

    ''' 
    SELECT INDIVIDUALS
    '''

    def select_individuals(self, population):
        # max_value, max_sample = self.find_max_sample(population) # esto es muy monobjetivo
        max_sample = self.find_max_sample_nds(
            population, self.nds)
        # max_sample = self.find_max_sample_pop(
        #    population)
        return max_sample

    def find_max_sample_monoscore(self, population):
        population.sort(
            key=lambda x: x.compute_mono_objective_score(), reverse=True)
        max_score = population[0].compute_mono_objective_score()
        max_sample = population[0]
        return max_score, max_sample

    def find_max_sample_nds(self, population, nds):
        if len(nds) > 0:
            random_index = np.random.randint(len(nds))
            return nds[random_index]
        else:
            random_index = np.random.randint(len(population))
            return population[random_index]

    def find_max_sample_pop(self, population):
        nds_pop = get_nondominated_solutions(population, [])
        #nds_pop = population
        random_index = np.random.randint(len(nds_pop))
        return nds_pop[random_index]

    ''' 
    LEARN PROBABILITY MODEL
    '''

    def learn_probability_model(self, probability_vector, max_sample):
        for i in np.arange(len(probability_vector)):
            probability_vector[i] = probability_vector[i]*(
                1-self.learning_rate)+max_sample.selected[i]*(self.learning_rate)

        for i in np.arange(len(probability_vector)):  # mutacion flip each bit
            prob = np.random.random_sample()
            if prob < self.mutation_prob:
                probability_vector[i] = probability_vector[i]*(
                    1-self.mutation_shift) + (np.random.randint(2))*self.mutation_shift
        return probability_vector

    ''' 
    SAMPLE NEW POPULATION
    '''

    def sample_new_population(self, probability_vector):
        new_population = []
        for i in np.arange(self.population_length):
            sample = self.generate_sample_from_probabilities(
                probability_vector)
            new_population.append(sample)
        return new_population

    def generate_sample_from_probabilities(self, probabilities):
        sample_selected = np.random.binomial(1, probabilities)

        sample = GraspSolution(self.dataset, None, selected=sample_selected)
        return sample


# RUN ALGORITHM------------------------------------------------------------------


    def run(self):
        start = time.time()
        self.reset()

        paretos = []

        returned_population = None
        self.probability_vector = self.initialize_probability_vector()

        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                self.population = []

                self.population = self.sample_new_population(self.probability_vector)

                # repair population if dependencies tackled:
                if(self.tackle_dependencies):
                    self.population = self.repair_population_dependencies(
                        self.population)

                self.evaluate(self.population, self.best_individual)

                max_sample = self.select_individuals(self.population)

                self.probability_vector = self.learn_probability_model(
                    self.probability_vector, max_sample)

                # update nds with solutions constructed and evolved in this iteration
                # self.update_nds(self.population)
                get_nondominated_solutions(self.population, self.nds)
                self.num_generations += 1
                if self.debug_mode:
                    paretos.append(self.nds)

        except EvaluationLimit:
            pass

        end = time.time()

        self.nds = format_population(self.nds, self.dataset)

        print("\nNDS created has", self.nds.__len__(), "solution(s)")

        return {"population": self.nds,
                "time": end - start,
                "numGenerations": self.num_generations,
                "best_individual": self.best_individual,
                "numEvaluations": self.num_evaluations,
                "paretos": paretos
                }
