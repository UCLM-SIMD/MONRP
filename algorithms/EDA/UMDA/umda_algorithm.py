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
                          f"{str(max_generations)}-{str(max_evaluations)}-{str(selected_individuals)}-{str(selection_scheme)}-"
                          f"{str(replacement_scheme)}.txt")

    def get_name(self) -> str:
        return (f"UMDA{str(self.population_length)}+{str(self.max_generations)}+{str(self.max_evaluations)}+"
                f"{str(self.selected_individuals)}+{str(self.selection_scheme)}+{str(self.replacement_scheme)}")

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
        # old_pop = []
        self.evaluate(self.population, self.best_individual)

        get_nondominated_solutions(self.population, self.nds)
        if self.debug_mode:
            paretos.append(self.nds.copy())

        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                # selection
                # TODO individuals = self.select_individuals(self.population+old_pop)
                individuals = self.select_individuals(self.population)
                    
                # learning
                probability_model = self.learn_probability_model(
                    individuals)

                # old_pop = self.population.copy()

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
                    paretos.append(self.nds.copy())

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
