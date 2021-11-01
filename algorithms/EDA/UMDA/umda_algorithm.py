from typing import Any, Dict, List
from algorithms.EDA.eda_algorithm import EDAAlgorithm
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from evaluation.get_nondominated_solutions import get_nondominated_solutions
from models.Solution import Solution
from algorithms.EDA.UMDA.umda_executer import UMDAExecuter

import time
import numpy as np


class UMDAAlgorithm(EDAAlgorithm):
    """Univariate Marginal Distribution Algorithm
    """

    def __init__(self, dataset_name: str = "test", random_seed: int = None, debug_mode: bool = False, tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0,
                 selected_individuals: int = 60, selection_scheme: str = "nds", replacement_scheme: str = "replacement"):

        super().__init__(dataset_name, random_seed, debug_mode, tackle_dependencies,
                         population_length, max_generations, max_evaluations)
        self.executer = UMDAExecuter(algorithm=self)

        self.selected_individuals: int = selected_individuals

        self.selection_scheme: str = selection_scheme
        self.replacement_scheme: str = replacement_scheme

        self.file: str = (f"{str(self.__class__.__name__)}-{str(dataset_name)}-{str(random_seed)}-{str(population_length)}-"
                          f"{str(max_generations)}-{str(max_evaluations)}.txt")

    def get_name(self) -> str:
        return (f"UMDA{str(self.population_length)}+{str(self.max_generations)}+"
                f"{str(self.max_evaluations)}")

    def learn_probability_model(self, population: List[Solution]) -> List[float]:
        """Learns probability from a set of solutions, returning an array of probabilities for each gene to be 1.
        """
        probability_model = []
        # for each gene:
        for index in np.arange(len(self.dataset.pbis_cost_scaled)):
            num_ones = 0
            # count selected
            for individual in population:
                num_ones += individual.selected[index]
            # prob = nº 1s / nº total
            index_probability = num_ones/len(population)
            probability_model.append(index_probability)

        return probability_model

    def generate_sample_from_probabilities_binomial(self, probabilities: List[float]) -> Solution:
        """Generates a sample given the probability vector, using numpy binomial method.
        """
        sample_selected = np.random.binomial(1, probabilities)
        sample = Solution(self.dataset, None, selected=sample_selected)
        return sample

    def generate_sample_from_probabilities(self, probabilities: List[float]) -> Solution:
        """Generates a sample given the probability vector, using scaled probabilities
        """
        probs = [prob * 10 for prob in probabilities]
        sum_probs = np.sum(probs)
        scaled_probs = probs / sum_probs
        sample = Solution(self.dataset, scaled_probs)
        return sample

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

    def replace_population_from_probabilities(self, probability_model: List[float]) -> List[Solution]:
        new_population = []
        for _ in np.arange(self.population_length):
            new_individual = self.generate_sample_from_probabilities_binomial(
                probability_model)
            # new_individual = self.generate_sample_from_probabilities(
            #    probability_model)
            new_population.append(new_individual)

        return new_population

    def sample_new_population(self, probability_model: List[float]) -> List[Solution]:
        """Given a probability vector, samples a new population depending on the scheme selected.
        """
        if self.replacement_scheme == "replacement":
            population = self.replace_population_from_probabilities(
                probability_model)
        elif self.replacement_scheme == "elitism":
            population = self.replace_population_from_probabilities_elitism(
                probability_model, self.population)
        return population

    def run(self) -> Dict[str, Any]:
        self.reset()
        paretos = []
        start = time.time()

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
                get_nondominated_solutions(self.population, self.nds)

                self.num_generations += 1

                if self.debug_mode:
                    paretos.append(self.nds)

        except EvaluationLimit:
            pass

        end = time.time()

        print("\nNDS created has", self.nds.__len__(), "solution(s)")

        return {"population": self.nds,
                "time": end - start,
                "numGenerations": self.num_generations,
                "best_individual": self.best_individual,
                "numEvaluations": self.num_evaluations,
                "paretos": paretos
                }
