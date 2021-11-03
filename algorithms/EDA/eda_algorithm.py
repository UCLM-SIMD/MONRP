from abc import abstractmethod
from typing import List
import numpy as np
from models.Solution import Solution
from algorithms.abstract_algorithm.abstract_algorithm import AbstractAlgorithm
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from evaluation.get_nondominated_solutions import get_nondominated_solutions


class EDAAlgorithm(AbstractAlgorithm):
    """Estimation of Distribution Algorithm
    """

    def __init__(self, dataset_name: str = "1", random_seed: int = None, debug_mode: bool = False, tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0,):

        super().__init__(dataset_name, random_seed, debug_mode, tackle_dependencies)

        self.nds = []
        self.num_evaluations: int = 0
        self.num_generations: int = 0
        self.best_individual = None

        self.population_length: int = population_length
        self.max_generations: int = max_generations
        self.max_evaluations: int = max_evaluations

    def generate_initial_population(self) -> List[Solution]:
        population = []
        candidates_score_scaled = np.full(
            self.dataset.pbis_score.size, 1/self.dataset.pbis_score.size)
        for _ in np.arange(self.population_length):
            ind = Solution(self.dataset, candidates_score_scaled)
            population.append(ind)
        return population

    def select_individuals(self, population: List[Solution]) -> List[Solution]:
        if self.selection_scheme == "nds":
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

    # @abstractmethod
    # def learn_probability_model(self):
    #     pass

    # @abstractmethod
    # def sample_new_population(self):
    #     pass

    def repair_population_dependencies(self, solutions: List[Solution]) -> List[Solution]:
        for sol in solutions:
            sol.correct_dependencies()
        return solutions

    def add_evaluation(self, new_population) -> None:
        self.num_evaluations += 1
        if (self.stop_criterion(self.num_generations, self.num_evaluations)):
            get_nondominated_solutions(new_population, self.nds)
            raise EvaluationLimit

    def reset(self) -> None:
        self.nds = []
        self.best_individual = None
        self.num_generations = 0
        self.num_evaluations = 0

    def stop_criterion(self, num_generations, num_evaluations) -> bool:
        if self.max_evaluations == 0:
            return num_generations >= self.max_generations
        else:
            return num_evaluations >= self.max_evaluations
