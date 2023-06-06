from typing import Any, Dict, List

import evaluation.solution_subset_selection
from algorithms.EDA.eda_algorithm import EDAAlgorithm
from algorithms.abstract_algorithm.abstract_algorithm import plot_solutions
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from datasets import Dataset
from evaluation.get_nondominated_solutions import get_nondominated_solutions
from models.Solution import Solution
from algorithms.EDA.UMDA.umda_executer import UMDAExecuter

import time
import numpy as np

from models.Hyperparameter import generate_hyperparameter


class UMDAAlgorithm(EDAAlgorithm):
    """Univariate Marginal Distribution Algorithm
    """

    def __init__(self, execs,dataset_name: str = "test", dataset: Dataset = None, random_seed: int = None, debug_mode: bool = False, tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0,
                 selected_individuals: int = 60, selection_scheme: str = "nds",
                 replacement_scheme: str = "replacement", subset_size: int = 20, sss_type=0, sss_per_it=False):

        super().__init__(execs,dataset_name, dataset, random_seed, debug_mode, tackle_dependencies,
                         population_length, max_generations, max_evaluations, subset_size=subset_size,
                         sss_type=sss_type, sss_per_iteration=sss_per_it)

        self.executer = UMDAExecuter(algorithm=self, execs=execs)

        self.selected_individuals: int = selected_individuals

        self.selection_scheme: str = selection_scheme
        self.replacement_scheme: str = replacement_scheme

        self.config_dictionary.update({'algorithm': 'umda'})

        self.hyperparameters.append(generate_hyperparameter(
            "selection_scheme", selection_scheme))
        self.config_dictionary['selection_scheme'] = selection_scheme
        self.hyperparameters.append(generate_hyperparameter(
            "replacement_scheme", replacement_scheme))
        self.config_dictionary['replacement_scheme'] = replacement_scheme

    def get_file(self) -> str:
        return (f"{str(self.__class__.__name__)}-{str(self.dataset_name)}-"
                f"{self.dependencies_to_string()}-{str(self.random_seed)}-{str(self.population_length)}-"
                f"{str(self.max_generations)}-{str(self.max_evaluations)}-{str(self.selected_individuals)}-{str(self.selection_scheme)}-"
                f"{str(self.replacement_scheme)}.txt")

    def get_name(self) -> str:
        return (f"UMDA{str(self.population_length)}+{str(self.max_generations)}+{str(self.max_evaluations)}+"
                f"{str(self.selected_individuals)}+{str(self.selection_scheme)}+{str(self.replacement_scheme)}")

    def df_find_data(self, df: any):
        return df[(df["Population Length"] == self.population_length) & (df["MaxGenerations"] == self.max_generations)
                  & (df["Selection Scheme"] == self.selection_scheme) & (df["Selected Individuals"] == self.selected_individuals)
                  & (df["Algorithm"] == self.__class__.__name__) & (df["Replacement Scheme"] == self.replacement_scheme)
                  & (df["Dataset"] == self.dataset_name) & (df["MaxEvaluations"] == self.max_evaluations)
                  ]

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
        start = time.time()
        nds_update_time = 0
        sss_total_time = 0

        self.population = self.generate_initial_population()
        if (self.tackle_dependencies):
            self.population = self.repair_population_dependencies(
                self.population)
        get_nondominated_solutions(self.population, self.nds)


        if self.debug_mode:
            self.debug_data()

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

                # evaluation  # update nds with solutions constructed and evolved in this iteration
                update_start = time.time()
                get_nondominated_solutions(self.population, self.nds)
                nds_update_time = nds_update_time + (time.time() - update_start)

                self.num_generations += 1

                if self.sss_per_iteration:
                    sss_start = time.time()
                    self.nds = evaluation.solution_subset_selection.search_solution_subset(self.sss_type,
                                                                                           self.subset_size, self.nds)
                    sss_total_time = sss_total_time + (time.time() - sss_start)


                if self.debug_mode:
                    self.debug_data()

        except EvaluationLimit:
            pass

        end = time.time()
        #plot_solutions(self.nds)

        print("\nNDS created has", self.nds.__len__(), "solution(s)")

        return {"population": self.nds,
                "time": end - start,
                "nds_update_time": nds_update_time,
                "sss_total_time": sss_total_time,
                "numGenerations": self.num_generations,
                "best_individual": self.best_individual,
                "numEvaluations": self.num_evaluations,
                "nds_debug": self.nds_debug,
                "population_debug": self.population_debug
                }
