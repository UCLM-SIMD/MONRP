<<<<<<< HEAD
<<<<<<< HEAD

from typing import Any, Dict, List
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
import copy
from algorithms.abstract_algorithm.abstract_algorithm import AbstractAlgorithm, plot_solutions
from algorithms.GRASP.grasp_executer import GRASPExecuter
import time

import numpy as np
import random
from datasets import Dataset

from models.Solution import Solution
from evaluation.get_nondominated_solutions import get_nondominated_solutions

from models.Hyperparameter import generate_hyperparameter


class GRASP(AbstractAlgorithm):
    """
    __author__      = "Victor.PerezPiqueras@uclm.es" "Pablo.Bermejo@uclm.es"

=======
=======
import time

import dataset1
>>>>>>> 2fb57e2a (GRASP terminado)
from algorithms.GRASP.Dataset import Dataset
import numpy as np
import getopt
import sys

from algorithms.GRASP.GraspSolution import GraspSolution
from algorithms.genetic_nds import genetic_nds_utils
from algorithms.genetic_nds.genetic_nds_utils import GeneticNDSUtils
from models.gen import Gen
from models.individual import Individual
from models.population import Population
from models.problem import Problem


class GRASP:
    """
<<<<<<< HEAD
>>>>>>> a7741b90 (GRASP y clases necesarias creado hasta la fase de construcción inicial de soluciones)
=======
    __author__      = "Pablo.Bermejo@uclm.es"

>>>>>>> 2fb57e2a (GRASP terminado)
    A GRASP search with two phases:
      - create initial set of solutions, based on a ranking of probabilities computed from
          pbi scores (as given by self.dataset.pbis_score)
      - local search, to improve solutions based on their neighbourhood

<<<<<<< HEAD
    """

    def __init__(self,  execs: int, dataset_name: str = "test", dataset: Dataset = None, iterations: int = 20,
                 solutions_per_iteration: int = 10, max_evaluations: int = 0,
                 init_type: str = "stochastically", local_search_type: str = "best_first_neighbor_random",
                 path_relinking_mode: str = "None", seed: int = None,
                 debug_mode: bool = False, tackle_dependencies: bool = False, subset_size: int = 5):

        super().__init__(execs,dataset_name, dataset, seed, debug_mode, tackle_dependencies, subset_size=subset_size)

        self.executer = GRASPExecuter(algorithm=self, execs=execs)
        self.config_dictionary.update({'algorithm': 'GRASP'})

        self.iterations: int = iterations
        self.solutions_per_iteration: int = solutions_per_iteration
        self.max_evaluations: int = max_evaluations

        self.hyperparameters.append(generate_hyperparameter(
            "iterations", iterations))
        self.config_dictionary['iterations'] = iterations
        self.hyperparameters.append(generate_hyperparameter(
            "solutions_per_iteration", solutions_per_iteration))
        self.config_dictionary['solutions_per_iteration'] = solutions_per_iteration
        self.hyperparameters.append(generate_hyperparameter(
            "max_evaluations", max_evaluations))
        self.config_dictionary['max_evaluations'] = max_evaluations
        self.hyperparameters.append(generate_hyperparameter(
            "init_type", init_type))
        self.config_dictionary['init_type'] = init_type
        self.hyperparameters.append(generate_hyperparameter(
            "local_search_type", local_search_type))
        self.config_dictionary['local_search_type'] = local_search_type
        self.hyperparameters.append(generate_hyperparameter(
            "path_relinking_mode", path_relinking_mode))
        self.config_dictionary['path_relinking_mode'] = path_relinking_mode

        self.nds: List[Solution] = []
        self.num_evaluations: int = 0
        self.num_iterations: int = 0
        self.start: int = 0

        self.init_type: str = init_type
        self.local_search_type: str = local_search_type
        self.path_relinking_mode: str = path_relinking_mode

        if self.init_type == "stochastically":
            self.initialize = self.init_solutions_stochastically
        elif self.init_type == "uniform":
            self.initialize = self.init_solutions_uniform

        if self.local_search_type == "best_first_neighbor_random":
            self.local_search = self.local_search_bitwise_neighborhood_random
        elif self.local_search_type == "best_first_neighbor_sorted_score":
            self.local_search = self.local_search_bitwise_neighborhood_sorted_score
        elif self.local_search_type == "best_first_neighbor_sorted_score_r":
            self.local_search = self.local_search_bitwise_neighborhood_sorted_score_r
        elif self.local_search_type == "best_first_neighbor_random_domination":
            self.local_search = self.local_search_bitwise_neighborhood_random_domination
        elif self.local_search_type == "best_first_neighbor_sorted_domination":
            self.local_search = self.local_search_bitwise_neighborhood_sorted_domination
        elif self.local_search_type == "None":
            self.local_search = "None"




    def get_file(self) -> str:
        return (f"{str(self.__class__.__name__)}-{str(self.dataset_name)}-"
                f"{self.dependencies_to_string()}-{str(self.random_seed)}-"
                f"{str(self.iterations)}-{str(self.max_evaluations)}-"
                f"{str(self.solutions_per_iteration)}-{str(self.init_type)}-"
                f"{self.local_search_type}-{str(self.path_relinking_mode)}.txt")

    def get_name(self) -> str:
        init = "stochastic" if self.init_type == "stochastically" else self.init_type
        local = self.local_search_type.replace(
            'best_first_neighbor_', '') if self.local_search_type != "None" else ""
        PR = "PR" if self.path_relinking_mode != "None" else ""
        return (f"GRASP+{str(self.iterations)}+{str(self.solutions_per_iteration)}+"
                f"{str(self.max_evaluations)}+{init}+{local}+{PR}")

    def df_find_data(self, df: any):
        return df[(df["Iterations"] == self.iterations) & (df["Solutions per Iteration"] == self.solutions_per_iteration)
                  & (df["Initialization Type"] == self.init_type) & (df["Local Search Type"] == self.local_search_type)
                  & (df["Path Relinking"] == self.path_relinking_mode) & (df["Algorithm"] == self.__class__.__name__)
                  & (df["Dataset"] == self.dataset_name) & (df["MaxEvaluations"] == self.max_evaluations)
                  ]

    def reset(self) -> None:
        super().reset()
        self.nds = []
        self.num_evaluations = 0
        self.num_iterations = 0
        self.start = 0

    def stop_criterion(self, num_iterations, num_evaluations) -> bool:
        if self.max_evaluations == 0:
            return num_iterations >= self.iterations
        else:
            return num_evaluations >= self.max_evaluations

    def add_evaluation(self, initiated_solutions) -> None:
        self.num_evaluations += 1
        if self.stop_criterion(self.num_iterations, self.num_evaluations):
            get_nondominated_solutions(initiated_solutions, self.nds)
            raise EvaluationLimit

    def run(self) -> Dict[str, Any]:
        """
        Core code of GRASP: initiation + local search + NDS update, repeated self.iterations times.
        :return (selected_list, seconds) list of ndarray and double.
                shape of selected is (len(self.nds), Solution.dataset.num_pbis)
                    position ij==0 if solution i does not select candidate j
                    position ij==1 if solution i selects candidate j
                seconds is the time in seconds used to run all the GRASP iterations

        """
        self.reset()
        self.start = time.time()

        self.num_iterations = 0
        try:
            while (not self.stop_criterion(self.num_iterations, self.num_evaluations)):
                # construction phase
                initiated_solutions = self.initialize()
                #get_nondominated_solutions(initiated_solutions, self.nds)
                #plot_solutions(initiated_solutions)

                # local search phase
                if self.local_search != "None":
                    initiated_solutions = self.local_search(
                        initiated_solutions)
                #plot_solutions(initiated_solutions)

                if self.path_relinking_mode == "after_local":
                    initiated_solutions = self.path_relinking(
                        initiated_solutions)
                #plot_solutions(initiated_solutions)

                # repair population if dependencies tackled:
                #plot_solutions(initiated_solutions)
                if(self.tackle_dependencies):
                    initiated_solutions = self.repair_population_dependencies(
                        initiated_solutions)
                #plot_solutions(initiated_solutions)
                # update NDS with solutions constructed and evolved in this iteration
                get_nondominated_solutions(initiated_solutions, self.nds)
                #plot_solutions(self.nds)

                self.num_iterations += 1

                if self.debug_mode:
                    self.debug_data()

        except EvaluationLimit:
            pass

        seconds = time.time() - self.start
        print("\nNDS created has", self.nds.__len__(), "solution(s)")
        #plot_solutions(self.nds)
        return {
            "population": self.nds,
            "time": seconds,
            "numGenerations": self.num_iterations,
            "numEvaluations": self.num_evaluations,
            "nds_debug": self.nds_debug,
            "population_debug": self.population_debug
        }

    def init_solutions_stochastically(self) -> List[Solution]:
        """
        candidates (pbis) are selected stochastically based on a rankin of the score of each pbi
        the ranking is scaled with values that sum up to 1. Each value is used as the probability to be chosen.
        :return solutions: list of GraspSolution
=======
    Attributes
    ----------
    dataset: type algorithms.GRASP.Dataset
        each pbi in Dataset is a candidate to be included in one of the GRASP solutions.
        Dataset provides cost, satisfaction and score of each pbi.

    solutions_per_iteration: integer (default = 10)
        number of GRASPSolutions to be initially created and evolved in local search

    NDS: list of algorithms.GRASP.GraspSolution, empty at start.
        it contains the current set of non dominated solutions

    local_search: string, type of search to perform.
        possible values are: best_first_neighbor (default)

    Methods
    ---------
    run(): the method which starts the search.
     It calls the two phases of GRASP: initiate solutions and evolve them with local search

    init_solutions(): stochastically construct a former set of solutions,
     each of them containing a set of chosen candidates (pbis)

    local_search_bitwise_neighborhood(initiated_solutions): evolves each of the constructed solutions in the initiation
    phase, running an incremental best first search over each solution, using mono_objective_score as goodness metric

    update_nds(solutions): at the end of each GRASP iteration, the global self.NDS list of solutions is updated based on the
        constructed and evolved solutions in such iteration.

    """

    def __init__(self, dataset=1, iterations=20, solutions_per_iteration=10,
                 local_search_type="best_first_neighbor", seed=1):
        """
        :param dataset: integer number: 1 or 2
        :param iterations: integer (default 20), number of GRASP construct+local_search repetitions
        :param solutions_per_iteration: number of GRASPSolutions to be initially created and evolved in local search
        :param local_search_type: type of search to evolve initiated solutions
        :param seed: int. seed for random generation of solutions in the first phase of each GRASP iteration
        """
        self.dataset = Dataset(dataset)
        self.iterations = iterations
        self.number_of_solutions = solutions_per_iteration
        self.NDS = []
        self.local_search = local_search_type
        self.seed = seed

    def run(self):
        """
        Core code of GRASP: initiation + local search + NDS update, repeated self.iterations times.
        :return (selected_list, seconds) list of ndarray and double.
                shape of selected is (len(self.NDS), GraspSolution.dataset.num_pbis)
                    position ij==0 if solution i does not select candidate j
                    position ij==1 if solution i selects candidate j
                seconds is the time in seconds used to run all the GRASP iterations

        """
        start = time.time()

        for _ in np.arange(self.iterations):
            # construction phase
            initiated_solutions = self.init_solutions()

            # local search phase
            if self.local_search == "best_first_neighbor":
                initiated_solutions = self.local_search_bitwise_neighborhood(initiated_solutions)

            # update NDS with solutions constructed and evolved in this iteration
            self.update_nds(initiated_solutions)

        seconds = time.time() - start

        print("\nNDS created has", self.NDS.__len__(), "solution(s)")
        selected_list = []
        for sol in self.NDS:
            #   print("\n-----------------------")
            #   print(sol)
            selected_list.append(sol.selected)
        # return selected_list, seconds

        return _results_in_victor_format(selected_list, seconds, self.iterations)

    def init_solutions(self):
        """
        candidates (pbis) are selected stochastically based on a rankin of the score of each pbi
        the ranking is scaled with values that sum up to 1. Each value is used as the probability to be chosen.
<<<<<<< HEAD
        :return:
>>>>>>> a7741b90 (GRASP y clases necesarias creado hasta la fase de construcción inicial de soluciones)
=======
        :return solutions: list of GraspSolution
>>>>>>> 2fb57e2a (GRASP terminado)
        """
        # scale candidates score, sum of scaled values is 1.0
        candidates_score_scaled = self.dataset.pbis_score / self.dataset.pbis_score.sum()

<<<<<<< HEAD
        solutions = []
        for i in np.arange(self.solutions_per_iteration):
            sol = Solution(self.dataset, candidates_score_scaled)
            # avoid solution with 0 cost due to 0 candidates selected
            if np.count_nonzero(sol.selected) > 0:
                solutions.append(sol)
            else:
                i -= 1
        return solutions

    def init_solutions_uniform(self) -> List[Solution]:
        """
        candidates (pbis) are selected uniformly 
        :return solutions: list of GraspSolution
        """
        # scale candidates score, sum of scaled values is 1.0
        candidates_score_scaled = np.full(
            self.dataset.pbis_score.size, 1/self.dataset.pbis_score.size)

        solutions = []
        for i in np.arange(self.solutions_per_iteration):
            sol = Solution(self.dataset, candidates_score_scaled)
            # avoid solution with 0 cost due to 0 candidates selected
            if np.count_nonzero(sol.selected) > 0:
                solutions.append(sol)
            else:
                i -= 1
        return solutions

    def local_search_bitwise_neighborhood_sorted_score(self, initiated_solutions: List[Solution]) -> List[Solution]:
        """
        For each initial solution, it runs an incremental search over the set of candidates, adding or removing each of them.
        Each time this operation improves sol.mono_objective_score, that change in sol is kept.
        Thus, this is an incremental best first search over each one of the solutions created in the initiation phase.
        The candidates of edach solution are sorted by score.

        for each sol in initiated_solutions
            for each candidate i
                add or remove that candidate i to find a neighbor of sol
                if neighbor has > mono_objective_score, then overwrite former solution with this neighbor
        :param initiated_solutions: list of GRASPSolution
        :return evolved version of initiated_solutions (same memory reference)

        """
        # https://stackoverflow.com/questions/9007877/sort-arrays-rows-by-another-array-in-python
        scores = self.dataset.pbis_score
        arr = np.arange(self.dataset.num_pbis)
        arr1inds = scores.argsort()
        sorted_pbis = arr[arr1inds[::-1]]

        for sol in initiated_solutions:
            for i in sorted_pbis:
                mo1 = sol.mono_objective_score
                # compute new cost, satisfaction and mo_score flipping candidate i in sol
                (c2, s2, mo2) = sol.try_flip(i, self.dataset.pbis_cost_scaled[i],
                                             self.dataset.pbis_satisfaction_scaled[i])
                self.add_evaluation(initiated_solutions)

                # if neighbor has greater mono_objective_score, then overwrite former solution with neighbor
                if mo2 > mo1 and np.count_nonzero(
                        sol.selected) > 0:  # avoid solution with 0 cost due to 0 candidates selected
                    sol.flip(
                        i, self.dataset.pbis_cost_scaled[i], self.dataset.pbis_satisfaction_scaled[i])
                    self.add_evaluation(initiated_solutions)


        return initiated_solutions

    def local_search_bitwise_neighborhood_sorted_score_r(self, initiated_solutions: List[Solution]) -> List[Solution]:
        """
        For each initial solution, it runs an incremental search over the set of candidates, adding or removing each of them.
        Each time this operation improves sol.mono_objective_score, that change in sol is kept. If not, it tries all the 
        combinations of removing a selected candidate, getting the best solution and replacing it if it improves the former 
        sol.mono_objective_score.
        The candidates of edach solution are first sorted by score.

        sort candidates
        for each sol in initiated_solutions
            for each candidate i
                add or remove that candidate i to find a neighbor of sol
                if neighbor has > mono_objective_score, then overwrite former solution with this neighbor
                else if neighbor has < mono_objective_score, then
                    for each other selected candidate 
                        try to remove it and save solution if new mono_objective_score>mono_objective_score
                    if any solution is better
                        replace solution by new solution

        :param initiated_solutions: list of GRASPSolution
        :return evolved version of initiated_solutions (same memory reference)

        """
        # https://stackoverflow.com/questions/9007877/sort-arrays-rows-by-another-array-in-python
        scores = self.dataset.pbis_score
        arr = np.arange(self.dataset.num_pbis)
        arr1inds = scores.argsort()
        sorted_pbis = arr[arr1inds[::-1]]

        for sol in initiated_solutions:
            for i in sorted_pbis:
                mo1 = sol.mono_objective_score
                # compute new cost, satisfaction and mo_score flipping candidate i in sol
                (c2, s2, mo2) = sol.try_flip(i, self.dataset.pbis_cost_scaled[i],
                                             self.dataset.pbis_satisfaction_scaled[i])
                self.add_evaluation(initiated_solutions)

                # if neighbor has greater mono_objective_score, then overwrite former solution with neighbor
                if mo2 > mo1 and np.count_nonzero(
                        sol.selected) > 0:  # avoid solution with 0 cost due to 0 candidates selected
                    sol.flip(
                        i, self.dataset.pbis_cost_scaled[i], self.dataset.pbis_satisfaction_scaled[i])
                    self.add_evaluation(initiated_solutions)
                elif mo2 <= mo1 and np.count_nonzero(sol.selected) > 0:
                    # get selected pbi indexes
                    selected_pbis_indexes = np.where(sol.selected == 1)[
                        0]  # no va squeeze
                    # for each one of them, try to flip and save the score
                    improving_indexes = []
                    improving_scores = []
                    for j in selected_pbis_indexes:
                        (c3, s3, mo3) = sol.try_flip(j, self.dataset.pbis_cost_scaled[j],
                                                     self.dataset.pbis_satisfaction_scaled[j])
                        self.add_evaluation(initiated_solutions)
                        if mo3 > mo1 and np.count_nonzero(
                                sol.selected) > 0:
                            improving_indexes.append(j)
                            improving_scores.append(mo3)
                    #  if there are better solutions, get index of best by score and replace solution
                    if len(improving_indexes) > 0:
                        max_mo_index = improving_indexes[np.argmax(
                            improving_scores)]
                        sol.flip(
                            max_mo_index, self.dataset.pbis_cost_scaled[max_mo_index], self.dataset.pbis_satisfaction_scaled[max_mo_index])
                        self.add_evaluation(initiated_solutions)

        return initiated_solutions

    def local_search_bitwise_neighborhood_random(self, initiated_solutions: List[Solution]) -> List[Solution]:
        """
        For each initial solution, it runs an incremental search over the set of candidates, adding or removing each of them.
        The order of change in the pbis is random.
        Each time this operation improves sol.mono_objective_score, that change in sol is kept.
        Thus, this is an incremental best first search over each one of the solutions created in the initiation phase

        for each sol in initiated_solutions
            generate candidate random ordering
            for each candidate i
                add or remove that candidate i to find a neighbor of sol
                if neighbor has > mono_objective_score, then overwrite former solution with this neighbor
        :param initiated_solutions: list of GRASPSolution
        :return evolved version of initiated_solutions (same memory reference)

        """
        for sol in initiated_solutions:
            arr = np.arange(self.dataset.num_pbis)
            np.random.shuffle(arr)
            for i in arr:
                mo1 = sol.mono_objective_score
                # compute new cost, satisfaction and mo_score flipping candidate i in sol
                (c2, s2, mo2) = sol.try_flip(i, self.dataset.pbis_cost_scaled[i],
                                             self.dataset.pbis_satisfaction_scaled[i])
                self.add_evaluation(initiated_solutions)
                #plot_solutions(initiated_solutions)

                # if neighbor has greater mono_objective_score, then overwrite former solution with neighbor
                if mo2 > mo1 and np.count_nonzero(
                        sol.selected) > 0:  # avoid solution with 0 cost due to 0 candidates selected
                    sol.flip(
                        i, self.dataset.pbis_cost_scaled[i], self.dataset.pbis_satisfaction_scaled[i])
                    self.add_evaluation(initiated_solutions)
            #plot_solutions(initiated_solutions)

        return initiated_solutions

    def local_search_bitwise_neighborhood_random_domination(self, initiated_solutions: List[Solution]) -> List[Solution]:
        """
        For each initial solution, it runs an incremental search over the set of candidates, adding or removing each of them.
        Each time this operation improves sol.mono_objective_score, that change in sol is kept.
        Thus, this is an incremental best first search over each one of the solutions created in the initiation phase.
        The candidates of edach solution are sorted by score.

        for each sol in initiated_solutions
            for each candidate i
                add or remove that candidate i to find a neighbor of sol
                if neighbor has > mono_objective_score, then overwrite former solution with this neighbor
        :param initiated_solutions: list of GRASPSolution
        :return evolved version of initiated_solutions (same memory reference)

        """

        for sol in initiated_solutions:
            arr = np.arange(self.dataset.num_pbis)
            np.random.shuffle(arr)
            for i in arr:
                # compute new cost, satisfaction and mo_score flipping candidate i in sol
                (c2, s2, mo2) = sol.try_flip(i, self.dataset.pbis_cost_scaled[i],
                                             self.dataset.pbis_satisfaction_scaled[i])
                self.add_evaluation(initiated_solutions)

                # if neighbor has greater mono_objective_score, then overwrite former solution with neighbor
                if sol.is_dominated_by_value(c2, s2) and np.count_nonzero(
                        sol.selected) > 0:  # avoid solution with 0 cost due to 0 candidates selected
                    sol.flip(
                        i, self.dataset.pbis_cost_scaled[i], self.dataset.pbis_satisfaction_scaled[i])
                    self.add_evaluation(initiated_solutions)

        return initiated_solutions

    def local_search_bitwise_neighborhood_sorted_domination(self, initiated_solutions: List[Solution]) -> List[Solution]:
        """
        For each initial solution, it runs an incremental search over the set of candidates, adding or removing each of them.
        Each time this operation improves sol.mono_objective_score, that change in sol is kept.
        Thus, this is an incremental best first search over each one of the solutions created in the initiation phase.
        The candidates of edach solution are sorted by score.

        for each sol in initiated_solutions
            for each candidate i
                add or remove that candidate i to find a neighbor of sol
                if neighbor has > mono_objective_score, then overwrite former solution with this neighbor
        :param initiated_solutions: list of GRASPSolution
        :return evolved version of initiated_solutions (same memory reference)

        """
        scores = self.dataset.pbis_score
        arr = np.arange(self.dataset.num_pbis)
        arr1inds = scores.argsort()
        sorted_pbis = arr[arr1inds[::-1]]
        for sol in initiated_solutions:
            for i in sorted_pbis:
                # compute new cost, satisfaction and mo_score flipping candidate i in sol
                (c2, s2, mo2) = sol.try_flip(i, self.dataset.pbis_cost_scaled[i],
                                             self.dataset.pbis_satisfaction_scaled[i])
                self.add_evaluation(initiated_solutions)

                # if neighbor has greater mono_objective_score, then overwrite former solution with neighbor
                if sol.is_dominated_by_value(c2, s2) and np.count_nonzero(
                        sol.selected) > 0:  # avoid solution with 0 cost due to 0 candidates selected
                    sol.flip(
                        i, self.dataset.pbis_cost_scaled[i], self.dataset.pbis_satisfaction_scaled[i])
                    self.add_evaluation(initiated_solutions)

        return initiated_solutions

    def path_relinking(self, solutions: List[Solution]) -> List[Solution]:
        """Method that, for each solution given, tries to find a path between it and a random solution of the nds set.
        At each step of the path, selects the solution that improves the monoscore value. At the end, the best solution found
        is stored in a new set of discovered solutions.
        """
        new_sols = []
        if len(self.nds) > 0:
            for solution in solutions:
                new_sols_path = []
                # get random solution from non dominated set
                random_nds_solution = random.choice(self.nds)
                # calculate distance from solution to goal random solution
                init_sol = copy.deepcopy(solution)
                distance = np.count_nonzero(
                    solution.selected != random_nds_solution.selected)
                # while distance greater than 0
                while(distance > 0):
                    mono_score = solution.compute_mono_objective_score()
                    # calculate indexes of different bits
                    diff_bits = np.where(solution.selected !=
                                         random_nds_solution.selected)
                    diff_bits = diff_bits[0]

                    # random shuffle bits to flip
                    np.random.shuffle(diff_bits)

                    # for each different bit, try flip and store the best if improves monoscore
                    best_mo = 0
                    selected_flip = None
                    for i in diff_bits:
                        (c2, s2, mo2) = init_sol.try_flip(i, self.dataset.pbis_cost_scaled[i],
                                                          self.dataset.pbis_satisfaction_scaled[i])
                        if mo2 > best_mo:
                            best_mo = mo2
                            selected_flip = i

                    # if it does not improves monoscore: select random
                    if mono_score >= best_mo:
                        selected_flip = np.random.choice(diff_bits)

                    # flip selected bit (best by monobjective or random) of the solution (copy created to not replace former)
                    init_sol.flip(
                        selected_flip, self.dataset.pbis_cost_scaled[selected_flip], self.dataset.pbis_satisfaction_scaled[selected_flip])
                    self.add_evaluation(solutions+new_sols)
                    distance = distance - 1

                    # save intermediate solution of the path
                    new_sols_path.append(init_sol)

                    # TODO actualizar NDS con soluciones intermedias(?)
                    # new_sols.append(solution)

                # after ending, select best path solution by monoscore
                if len(new_sols_path) > 0:
                    best_sol_path = max(
                        new_sols_path, key=lambda x: x.compute_mono_objective_score())
                    new_sols.append(best_sol_path)
                # best_sol_path=new_sols_path[0]

                # append it to new list of solutions

            # add best path solutions to old solution list
            solutions += new_sols

        return solutions




=======
        # create GraspSolutions
        np.random.seed(self.seed)
        solutions = []
        for i in np.arange(self.number_of_solutions):
            sol = GraspSolution(candidates_score_scaled, costs=self.dataset.pbis_cost_scaled,
                                values=self.dataset.pbis_satisfaction_scaled)
            if np.count_nonzero(sol.selected) > 0:  # avoid solution with 0 cost due to 0 candidates selected
                solutions.append(sol)
                i -= 1
        return solutions

    def local_search_bitwise_neighborhood(self, initiated_solutions):
        """
        For each initial solution, it runs an incremental search over the set of candidates, adding or removing each of them.
        Each time this operation improves sol.mono_objective_score, that change in sol is kept.
        Thus, this is an incremental best first search over each one of the solutions created in the initiation phase

        for each sol in initiated_solutions
            for each candidate i
                add or remove that candidate i to find a neighbor of sol
                if neighbor has > mono_objective_score, then overwrite former solution with this neighbor
        :param initiated_solutions: list of GRASPSolution
        :return evolved version of initiated_solutions (same memory reference)

        """

        for sol in initiated_solutions:
            for i in np.arange(self.dataset.num_pbis):
                # c1 = sol.total_cost
                # s1 = sol.total_satisfaction
                mo1 = sol.mono_objective_score
                # compute new cost, satisfaction and mo_score flipping candidate i in sol
                (c2, s2, mo2) = sol.try_flip(i, self.dataset.pbis_cost_scaled[i],
                                             self.dataset.pbis_satisfaction_scaled[i])

                # no uso la dominancia porque cambiando solo 1 bit nunca se crea una solución dominante!
                # if c2 < c1 and s2 > s1:  # if neighbor  dominates, then overwrite former solution with neighbor

                # if neighbor has greater mono_objective_score, then overwrite former solution with neighbor
                if mo2 > mo1 and np.count_nonzero(
                        sol.selected) > 0:  # avoid solution with 0 cost due to 0 candidates selected
                    sol.flip(i, self.dataset.pbis_cost_scaled[i], self.dataset.pbis_satisfaction_scaled[i])

        return initiated_solutions

    def update_nds(self, solutions):
        """
        For each sol in solutions:
            if no solution in self.NDS dominates sol:
             insert sol in self.NDS
             remove all solutions in self.NDS now dominated by sol
        :param solutions: solutions created in current GRASP iteration and evolved with local search
        """
        for sol in solutions:
            insert = True

            # find which solutions, if any, in self.NDS are dominated by sol
            # if sol is dominated by any solution in self.NDS, then search is stopped and sol is discarded
            now_dominated = []
            for nds_sol in self.NDS:

                if sol.dominates(nds_sol):
                    now_dominated.append(nds_sol)
                # left part of elif is because if some solution in NDS is already dominated by sol,
                # then no other solution in NDS will dominate sol, so evaluating it is waste
                elif (now_dominated.__len__ == 0 and nds_sol.dominates(sol)) \
                        or np.array_equal(sol.selected, nds_sol.selected):  # sol already existed in self.NDS
                    insert = False
                    break

            # sol is inserted if it is not dominated by any solution in self.NDS,
            # and all solutions in self.NDS dominated by sol are then removed
            if insert:
                self.NDS.append(sol)
                for dominated in now_dominated:
                    self.NDS.remove(dominated)


def _results_in_victor_format(nds, seconds, num_iterations):
    # convertir solución al formato de las soluciones de Victor
    genes = dataset1.generate_dataset1_genes()
    problem = Problem(genes, ["MAX", "MIN"])
    final_nds_formatted = []
    for solution in nds:
        individual = Individual(problem.genes, problem.objectives)
        for b in np.arange(len(individual.genes)):
            individual.genes[b].included = solution[b]
            individual.evaluate_fitness()
        final_nds_formatted.append(individual)
    # calcular métricas
    utils = GeneticNDSUtils(problem=problem, random_seed=1)  # aquí la semilla no sirve de nada
    avg_value = utils.calculate_avgValue(final_nds_formatted)
    best_avg_value = utils.calculate_bestAvgValue(final_nds_formatted)
    hv = utils.calculate_hypervolume(final_nds_formatted)
    spread = utils.calculate_spread(final_nds_formatted)
    num_solutions = utils.calculate_numSolutions(final_nds_formatted)
    spacing = utils.calculate_spacing(final_nds_formatted)
    return {
        "population": final_nds_formatted,
        "time": seconds,
        "best_individual": None,
        "avg_value": avg_value,
        "best_avg_value": best_avg_value,
        "hv": hv,
        "spread": spread,
        "numSolutions": num_solutions,
        "spacing": spacing,
        "best_generation_num": None,
        "num_generations": num_iterations,
    }


def _get_options(argv=None):
    """out of class method used to gather dataset options
        :param argv: string with list of args and values
        :return: all the options
    """

    dataset = 1
    iterations = 20
    solutions_per_iteration = 10
    local_search_type = "best_first_neighbor"
    seed = 1
    try:
        opts, args = getopt.getopt(argv, "d:i:s:l:S:", ['dataset=', 'iterations=', 'solutions_per_it=', 'local_search=',
                                                        'seed='])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-d", "--dataset"):
            dataset = int(arg)
        elif opt in ("-i", "--iterations"):
            iterations = int(arg)
        elif opt in ("-s", "--solutions_per_it"):
            solutions_per_iteration = int(arg)
        elif opt in ("-l", "--local_search"):
            local_search_type = arg
        elif opt in ("-S", "--seed"):
            seed = int(arg)

    return dataset, iterations, solutions_per_iteration, local_search_type, seed


if __name__ == "__main__":
<<<<<<< HEAD
    g = GRASP(dataset=1, number_of_solutions=5)
    g.run()
>>>>>>> a7741b90 (GRASP y clases necesarias creado hasta la fase de construcción inicial de soluciones)
=======
    d, it, sols_per_it, search, s = _get_options(sys.argv[1:])
    g = GRASP(dataset=d, iterations=it, solutions_per_iteration=sols_per_it, local_search_type=search, seed=s)
    print("dataset:", d, "\niterations:", it, "\nsolutions_per_iteration:", sols_per_it, "\nlocal_search_type:",
          search, "\nseed:", s)

    results = g.run()
    for y in results:
        print(y, ':', results[y])
>>>>>>> 2fb57e2a (GRASP terminado)
