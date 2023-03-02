from typing import Any, Dict, List
import numpy as np

import evaluation
from algorithms.abstract_algorithm.abstract_algorithm import AbstractAlgorithm
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
import time
from algorithms.random.random_executer import RandomExecuter
from datasets import Dataset
from evaluation.get_nondominated_solutions import get_nondominated_solutions
from models.Solution import Solution


class RandomAlgorithm(AbstractAlgorithm):
    """Random algorithm that generates random solutions and updates a NDS set.
    """

    def __init__(self, execs, dataset_name: str = "test", dataset: Dataset = None, random_seed: int = None,
                 debug_mode: bool = False, tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0,
                 subset_size: int = 20, sss_type=0, sss_per_it=False):

        super().__init__(execs, dataset_name, dataset, random_seed, debug_mode, tackle_dependencies,
                         subset_size=subset_size, sss_type=sss_type, sss_per_iteration=sss_per_it)

        self.population_length: int = population_length
        self.max_generations: int = max_generations
        self.max_evaluations: int = max_evaluations

        self.nds = []
        self.num_evaluations: int = 0
        self.num_generations: int = 0

        self.executer = RandomExecuter(algorithm=self, execs=execs)
        self.config_dictionary.update({'algorithm': 'random'})

        self.config_dictionary['population_length'] = population_length
        self.config_dictionary['max_generations'] = max_generations
        self.config_dictionary['max_evaluations'] = max_evaluations

    def get_file(self) -> str:
        return (f"{str(self.__class__.__name__)}-{str(self.dataset_name)}-"
                f"{self.dependencies_to_string()}-{str(self.random_seed)}-{str(self.population_length)}-"
                f"{str(self.max_generations)}-{str(self.max_evaluations)}.txt")

    def get_name(self) -> str:
        return f"Random{str(self.population_length)}+{str(self.max_generations)}+{str(self.max_evaluations)}"

    def reset(self) -> None:
        """Specific reset implementation
        """
        super().reset()
        self.nds = []
        self.num_generations = 0
        self.num_evaluations = 0

    # RUN ALGORITHM------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        self.reset()
        start = time.time()
        nds_update_time = 0
        sss_total_time = 0

        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                # selection
                self.population = self.init_solutions_uniform(
                    self.population_length)

                # repair population if dependencies tackled:
                if (self.tackle_dependencies):
                    self.population = self.repair_population_dependencies(
                        self.population)

                # update nds with solutions constructed and evolved in this iteration
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

        # plot_solutions(self.nds)

        return {"population": self.nds,
                "time": end - start,
                "nds_update_time": nds_update_time,
                "sss_total_time": sss_total_time,
                "numGenerations": self.num_generations,
                "best_individual": self.nds[np.random.randint(low=0, high=len(self.nds))],
                "numEvaluations": self.num_evaluations,
                "nds_debug": self.nds_debug,
                "population_debug": self.population_debug
                }

    def init_solutions_uniform(self, population_length: int) -> List[Solution]:
        """
        candidates (pbis) are selected uniformly 
        :return solutions: list of GraspSolution
        """
        # scale candidates score, sum of scaled values is 1.0
        candidates_score_scaled = np.full(
            self.dataset.pbis_score.size, 1/self.dataset.pbis_score.size)

        solutions = []
        for i in np.arange(population_length):
            sol = Solution(self.dataset, candidates_score_scaled)
            # avoid solution with 0 cost due to 0 candidates selected
            if np.count_nonzero(sol.selected) > 0:
                solutions.append(sol)
            else:
                i -= 1
        return solutions

    def add_evaluation(self, new_population) -> None:

        return None

    def stop_criterion(self, num_generations, num_evaluations) -> bool:
        if self.max_evaluations == 0:
            return num_generations >= self.max_generations
        else:
            return num_evaluations >= self.max_evaluations
