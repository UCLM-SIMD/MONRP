from abc import abstractmethod
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from models.Solution import Solution
from algorithms.abstract_algorithm.abstract_algorithm import AbstractAlgorithm
import evaluation.metrics as metrics


class AbstractGeneticAlgorithm(AbstractAlgorithm):
    """Abstract class for genetic algorithms
    """

    def __init__(self, dataset_name: str = "test", random_seed: int = None, debug_mode: bool = False, tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0,
                 selection: str = "tournament", selection_candidates: int = 2, crossover: str = "onepoint", crossover_prob: float = 0.9,
                 mutation: str = "flipeachbit", mutation_prob: float = 0.1,
                 replacement: str = "elitism",):
        """Init method calls parent init and includes specific parameters of genetic algorithms
        """
        super().__init__(dataset_name, random_seed, debug_mode, tackle_dependencies)

        self.population_length: int = population_length
        self.max_generations: int = max_generations
        self.max_evaluations: int = max_evaluations

        self.selection_scheme: str = selection
        self.selection_candidates: int = selection_candidates
        self.crossover_scheme: str = crossover
        self.crossover_prob: float = crossover_prob
        self.mutation_scheme: str = mutation
        self.mutation_prob: float = mutation_prob
        self.replacement_scheme: str = replacement

        self.population = None
        self.best_generation_avgValue = None
        self.best_generation = None

        self.nds = []
        self.num_evaluations: int = 0
        self.num_generations: int = 0
        self.best_individual = None

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_file(self) -> str:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def df_find_data(self, df: any):
        return df[(df["Population Length"] == self.population_length) & (df["MaxGenerations"] == self.max_generations)
                  & (df["Selection Candidates"] == self.selection_candidates) & (df["Selection Scheme"] == self.selection_scheme)
                  & (df["Crossover Scheme"] == self.crossover_scheme) & (df["Crossover Probability"] == self.crossover_prob)
                  & (df["Mutation Scheme"] == self.mutation_scheme) & (df["Mutation Probability"] == self.mutation_prob)
                  & (df["Replacement Scheme"] == self.replacement_scheme) & (df["Algorithm"] == self.__class__.__name__)
                  & (df["Dataset"] == self.dataset_name) & (df["MaxEvaluations"] == self.max_evaluations)
                  ]

    def stop_criterion(self, num_generations, num_evaluations) -> bool:
        if self.max_evaluations == 0:
            return num_generations >= self.max_generations
        else:
            return num_evaluations >= self.max_evaluations

    def reset(self) -> None:
        super().reset()
        self.best_generation_avgValue = 0
        self.best_generation = 0
        self.num_evaluations = 0
        self.num_generations = 0
        self.best_individual = None
        self.population = None

    @abstractmethod
    def add_evaluation(self, new_population):
        pass

    def generate_starting_population(self) -> List[Solution]:
        """Method that generates a starting population of solutions
        """
        population = []
        for i in range(0, self.population_length):
            individual = Solution(self.dataset, None, uniform=True)
            population.append(individual)
        return population

    # LAST GENERATION ENHANCE------------------------------------------------------------------
    def calculate_last_generation_with_enhance(self, best_generation, best_generation_avgValue, num_generation, population) -> Tuple[int, float]:
        bestAvgValue = metrics.calculate_bestAvgValue(population)
        if bestAvgValue > best_generation_avgValue:
            best_generation_avgValue = bestAvgValue
            best_generation = num_generation
        return best_generation, best_generation_avgValue

    def crossover_one_point(self, population: List[Solution]) -> List[Solution]:
        """Default crossover operator
        """
        new_population = []
        i = 0
        while i < len(population):
            # if last element is alone-> add it
            if i == len(population) - 1:
                new_population.append(population[i])
            else:
                # pair 2 parents -> crossover or add them and jump 1 index extra
                prob = random.random()
                if prob < self.crossover_prob:
                    offsprings = self.crossover_aux_one_point(
                        population[i], population[i+1])
                    new_population.extend(offsprings)
                else:
                    new_population.extend(
                        [population[i], population[i+1]])
                i += 1
            i += 1
        return new_population

    def crossover_aux_one_point(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """Crossover aux method
        """
        chromosome_length = len(parent1.selected)
        # index aleatorio del punto de division para el cruce
        crossover_point = random.randint(1, chromosome_length - 1)

        offspring_genes1 = np.concatenate((parent1.selected[0:crossover_point],
                                           parent2.selected[crossover_point:]))
        offspring_genes2 = np.concatenate((parent2.selected[0:crossover_point],
                                           parent1.selected[crossover_point:]))

        offspring1 = Solution(
            self.dataset, None, selected=offspring_genes1)
        offspring2 = Solution(
            self.dataset, None, selected=offspring_genes2)

        return offspring1, offspring2

    def mutation_flip1bit(self, population: List[Solution]) -> List[Solution]:
        """Default mutation operator
        """
        new_population = []
        new_population.extend(population)

        for individual in new_population:
            prob = random.random()
            if prob < self.mutation_prob:
                chromosome_length = len(individual.selected)
                mutation_point = random.randint(0, chromosome_length - 1)
                if individual.selected[mutation_point] == 0:
                    individual.set_bit(mutation_point, 1)
                else:
                    individual.set_bit(mutation_point, 0)

        return new_population

    def mutation_flipeachbit(self, population: List[Solution]) -> List[Solution]:
        """Default mutation operator
        """
        new_population = []
        new_population.extend(population)

        for individual in new_population:
            for gen_index in range(len(individual.selected)):
                prob = random.random()
                if prob < self.mutation_prob:
                    if individual.selected[gen_index] == 0:
                        individual.set_bit(gen_index, 1)
                    else:
                        individual.set_bit(gen_index, 0)

        return new_population
