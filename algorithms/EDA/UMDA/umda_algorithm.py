from algorithms.EDA.eda_algorithm import EDAAlgorithm
from algorithms.abstract_default.evaluation_exception import EvaluationLimit
from evaluation.update_nds import get_nondominated_solutions
from models.Solution import Solution
from algorithms.EDA.UMDA.umda_executer import UMDAExecuter

import time
import numpy as np


class UMDAAlgorithm(EDAAlgorithm):  # Univariate Marginal Distribution Algorithm
    def __init__(self, dataset_name:str="test", random_seed:int=None, debug_mode:bool=False, tackle_dependencies:bool=False,
                population_length:int=100, max_generations:int=100, max_evaluations:int=0,
                 selected_individuals:int=60, selection_scheme:str="nds", replacement_scheme:str="replacement"):


        super().__init__(dataset_name, random_seed, debug_mode, tackle_dependencies,
            population_length, max_generations, max_evaluations)
        self.executer = UMDAExecuter(algorithm=self)
        # self.problem, self.dataset = self.utils.generate_dataset_problem(
        #    dataset_name=dataset_name)

        #self.dataset = Dataset(dataset_name)
        #self.dataset_name = dataset_name

        #self.population_length = population_length
        #self.max_generations = max_generations
        #self.max_evaluations = max_evaluations

        self.selected_individuals:int = selected_individuals

        self.selection_scheme:str = selection_scheme
        self.replacement_scheme:str = replacement_scheme

        #self.nds = []
        #self.num_evaluations = 0
        #self.num_generations = 0
        #self.best_individual = None
        #self.debug_mode = debug_mode
        #self.tackle_dependencies = tackle_dependencies

        # TODO los utils no se usan y estan mal los super()

        #self.random_seed = random_seed
        #if random_seed is not None:
        #    np.random.seed(random_seed)

        self.file:str = str(self.__class__.__name__)+"-"+str(dataset_name)+"-"+str(random_seed)+"-"+str(population_length)+"-" +\
            str(max_generations) + "-"+str(max_evaluations)+".txt"

    def get_name(self):
        return f"UMDA selection{self.selection_scheme} {self.replacement_scheme}"


    ''' 
    LEARN PROBABILITY MODEL
    '''

    def learn_probability_model(self, population):  # suavizar con laplace(?)
        probability_model = []
        # para cada gen:
        for index in np.arange(len(self.dataset.pbis_cost_scaled)):
            num_ones = 0
            # contar cuantos 1 hay en la poblacion
            for individual in population:
                num_ones += individual.selected[index]
            # prob = nº 1s / nº total
            index_probability = num_ones/len(population)
            probability_model.append(index_probability)

        return probability_model

    ''' 
    SAMPLE NEW POPULATION
    '''

    def generate_sample_from_probabilities_binomial(self, probabilities):
        #probs = probabilities/len(probabilities)
        sample_selected = np.random.binomial(1, probabilities)
        sample = Solution(self.dataset, None, selected=sample_selected)
        return sample

    def generate_sample_from_probabilities(self, probabilities):
        probs = [prob * 10 for prob in probabilities]
        sum_probs = np.sum(probs)
        scaled_probs = probs / sum_probs
        sample = Solution(self.dataset, scaled_probs)
        return sample

    def replace_population_from_probabilities_elitism(self, probability_model, population):
        new_population = []
        # elitist R-1 inds
        for i in np.arange(self.population_length-1):
            # new_individual = GraspSolution(self.dataset,probability_model, )
            #new_individual= np.random.choice([0, 1], size=len(self.dataset.pbis_cost_scaled), p=probability_model)
            new_individual = self.generate_sample_from_probabilities_binomial(
                probability_model)
            # new_individual = self.generate_sample_from_probabilities(
            #    probability_model)
            new_population.append(new_individual)

        # elitism -> add best individual from old population
        population.sort(
            key=lambda x: x.compute_mono_objective_score(), reverse=True)
        new_population.append(population[0])

        return new_population

    def replace_population_from_probabilities(self, probability_model):
        new_population = []
        for i in np.arange(self.population_length):
            new_individual = self.generate_sample_from_probabilities_binomial(
                probability_model)
            # new_individual = self.generate_sample_from_probabilities(
            #    probability_model)
            new_population.append(new_individual)

        return new_population

    def sample_new_population(self, probability_model):
        if self.replacement_scheme == "replacement":
            population = self.replace_population_from_probabilities(
                probability_model)
        elif self.replacement_scheme == "elitism":
            population = self.replace_population_from_probabilities_elitism(
                probability_model, self.population)
        return population

    # RUN ALGORITHM------------------------------------------------------------------

    def run(self):
        self.reset()
        paretos = []
        start = time.time()

        returned_population = None
        self.population = self.generate_initial_population()
        self.evaluate(self.population, self.best_individual)

        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                # selection
                individuals = self.select_individuals(self.population)

                # learning
                probability_model = self.learn_probability_model(
                    individuals)

                # replacement
                self.population = self.sample_new_population(probability_model)

                # repair population if dependencies tackled:
                if(self.tackle_dependencies):
                    self.population = self.repair_population_dependencies(
                        self.population)

                # evaluation
                self.evaluate(self.population, self.best_individual)

                # update nds with solutions constructed and evolved in this iteration
                #self.update_nds(self.population, self.nds)
                get_nondominated_solutions(self.population, self.nds)

                self.num_generations += 1

                if self.debug_mode:
                    paretos.append(self.nds)

        except EvaluationLimit:
            pass

        #self.nds = format_population(self.nds, self.dataset)
        end = time.time()

        print("\nNDS created has", self.nds.__len__(), "solution(s)")

        return {"population": self.nds,
                "time": end - start,
                "numGenerations": self.num_generations,
                "best_individual": self.best_individual,
                "numEvaluations": self.num_evaluations,
                "paretos": paretos
                }
