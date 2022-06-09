# from abc import abstractmethod
from typing import List
import numpy as np
from datasets import Dataset
from models.Solution import Solution
from algorithms.abstract_algorithm.abstract_algorithm import AbstractAlgorithm, plot_solutions
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from evaluation.get_nondominated_solutions import get_nondominated_solutions

from models.Hyperparameter import generate_hyperparameter


class EDAAlgorithm(AbstractAlgorithm):
    """Estimation of Distribution Algorithm
    """

    def __init__(self, execs: int, dataset_name: str = "1", dataset: Dataset = None, random_seed: int = None,
                 debug_mode: bool = False, tackle_dependencies: bool = False,
<<<<<<< HEAD
<<<<<<< HEAD
                 population_length: int = 100, max_generations: int = 100,
                 max_evaluations: int = 0, subset_size: int = 5):

        super().__init__(execs,dataset_name, dataset, random_seed, debug_mode, \
                         tackle_dependencies, subset_size = subset_size)
=======
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0,):

        super().__init__(execs,dataset_name, dataset, random_seed, debug_mode, tackle_dependencies)
>>>>>>> 19c7836f (ahora todos los resultados se almacenan en results.json con un id unico para cada conjunto de parametros de lanzamiento)
=======
                 population_length: int = 100, max_generations: int = 100,
                 max_evaluations: int = 0, subset_size: int = 5):

        super().__init__(execs,dataset_name, dataset, random_seed, debug_mode, \
                         tackle_dependencies, subset_size = subset_size)
>>>>>>> 5efa3a53 (new hyperparameter created: subset_size used to choose a subset of solutions from the final set of solutions returned by the executed algorithm. Also, nsgaii is added in extract_postMetrics.py.)

        self.nds = []
        self.num_evaluations: int = 0
        self.num_generations: int = 0
        self.best_individual = None

        self.population_length: int = population_length
        self.max_generations: int = max_generations
        self.max_evaluations: int = max_evaluations

        self.hyperparameters.append(generate_hyperparameter(
            "population_length", population_length))
        self.config_dictionary['population_length'] = population_length
        self.hyperparameters.append(generate_hyperparameter(
            "max_generations", max_generations))
        self.config_dictionary['max_generations'] = max_generations
        self.hyperparameters.append(generate_hyperparameter(
            "max_evaluations", max_evaluations))
        self.config_dictionary['max_evaluations'] = max_evaluations

    def generate_initial_population(self) -> List[Solution]:
        population = []
        candidates_score_scaled = np.full(
            self.dataset.pbis_score.size, 1/self.dataset.pbis_score.size)

        for _ in np.arange(self.population_length):
            ind = Solution(self.dataset, candidates_score_scaled)
            population.append(ind)
        #plot_solutions(population)
        return population

    def select_individuals(self, population: List[Solution]) -> List[Solution]:
<<<<<<< HEAD
<<<<<<< HEAD
        individuals = None
=======

>>>>>>> bd41d390 (first version of FEDA (Fixed-structure EDA) finished)
=======
        individuals = None
>>>>>>> 19c7836f (ahora todos los resultados se almacenan en results.json con un id unico para cada conjunto de parametros de lanzamiento)
        if self.selection_scheme == "nds":
            # TODO
            # if len(self.nds) > 0:
            #     individuals = self.select_nondominated_individuals(
            #         self.nds)
            # else:
            individuals = self.select_nondominated_individuals(
                population)

        elif self.selection_scheme == "monoscore":
            individuals = self.select_individuals_monoscore(population)
        return individuals

    def select_individuals_monoscore(self, population: List[Solution]) -> List[Solution]:
        individuals = []
        population.sort(
            key=lambda x: x.compute_mono_objective_score(), reverse=True)

        for i in np.arange(self.selected_individuals):
            individuals.append(population[i])

        return individuals

    def select_nondominated_individuals(self, population: List[Solution]) -> List[Solution]:
        selected_individuals = get_nondominated_solutions(population, [])
        return selected_individuals

    def generate_sample_from_probabilities_binomial(self, probabilities: List[float]) -> Solution:
        """Generates a sample given the probability vector, using numpy binomial method.
        """
        x = 0
        while(x <= 0):
            sample_selected = np.random.binomial(1, probabilities)
            x = np.count_nonzero(sample_selected)
        sample_selected = np.where(sample_selected == 1)
        sample = Solution(self.dataset, None, selected=sample_selected)
        return sample

    # def generate_sample_from_probabilities(self, probabilities: List[float]) -> Solution:
    #     """Generates a sample given the probability vector, using scaled probabilities
    #     """
    #     probs = [prob * 10 for prob in probabilities]
    #     sum_probs = np.sum(probs)
    #     scaled_probs = probs / sum_probs
    #     sample = Solution(self.dataset, scaled_probs)
    #     return sample

    def replace_population_from_probabilities(self, probability_model: List[float]) -> List[Solution]:
        new_population = []
        for _ in np.arange(self.population_length):
            new_individual = self.generate_sample_from_probabilities_binomial(
                probability_model)
            # new_individual = self.generate_sample_from_probabilities(
            #    probability_model)
            new_population.append(new_individual)

        return new_population

    def replace_population_from_probabilities_elitism(self, probability_model: List[float], population: List[Solution]) -> List[Solution]:
        new_population = []
        # elitist R-1 inds
        for _ in np.arange(self.population_length-1):
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

    # @abstractmethod
    # def learn_probability_model(self):
    #     pass

    # @abstractmethod
    # def sample_new_population(self):
    #     pass

    def add_evaluation(self, new_population) -> None:
        self.num_evaluations += 1
        if (self.stop_criterion(self.num_generations, self.num_evaluations)):
            get_nondominated_solutions(new_population, self.nds)
            raise EvaluationLimit

    def reset(self) -> None:
        super().reset()
        self.nds = []
        self.best_individual = None
        self.num_generations = 0
        self.num_evaluations = 0

    def stop_criterion(self, num_generations, num_evaluations) -> bool:
        if self.max_evaluations == 0:
            return num_generations >= self.max_generations
        else:
            return num_evaluations >= self.max_evaluations
