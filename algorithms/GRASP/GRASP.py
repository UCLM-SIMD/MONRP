from algorithms.abstract_default.evaluation_exception import EvaluationLimit
import copy
from algorithms.abstract_default.algorithm import Algorithm
from algorithms.GRASP.grasp_executer import GRASPExecuter
import time

import numpy as np
import getopt
import sys
import random

from models.Solution import Solution
from evaluation.update_nds import get_nondominated_solutions


class GRASP(Algorithm):
    """
    __author__      = "Victor.PerezPiqueras@uclm.es" "Pablo.Bermejo@uclm.es"

    A GRASP search with two phases:
      - create initial set of solutions, based on a ranking of probabilities computed from
          pbi scores (as given by self.dataset.pbis_score)
      - local search, to improve solutions based on their neighbourhood

    Attributes
    ----------
    dataset_name: string
        name of the dataset

    dataset: type algorithms.GRASP.Dataset
        each pbi in Dataset is a candidate to be included in one of the GRASP solutions.
        Dataset provides cost, satisfaction and score of each pbi.

    solutions_per_iteration: integer (default = 10)
        number of GRASPSolutions to be initially created and evolved in local search

    NDS: list of algorithms.GRASP.GraspSolution, empty at start.
        it contains the current set of non dominated solutions

    init_type: string, type of initialization.
        possible values are: stochastically (default), uniform

    local_search_type: string, type of search to perform.
        possible values are: best_first_neighbor_random (default), best_first_neighbor_sorted_score,
        best_first_neighbor_sorted_score_r, best_first_neighbor_random_domination, best_first_neighbor_sorted_domination

    path_relinking_mode: type of path relinking.
    possible values are: None (default), after_local

    Methods
    ---------
    get_name(): returns name of the algorithm.

    run(): the method which starts the search.
     It calls the two phases of GRASP: initiate solutions and evolve them with local search

    init_solutions_stochastically(): stochastically construct a former set of solutions,
     each of them containing a set of chosen candidates (pbis)

    init_solutions_uniform(): uniformly construct a former set of solutions,
     each of them containing a set of chosen candidates (pbis)

    local_search_bitwise_neighborhood_random(initiated_solutions): evolves each of the constructed solutions in the initiation
    phase, running an incremental best first search over each solution, using mono_objective_score as goodness metric and randomness

    local_search_bitwise_neighborhood_sorted_score(initiated_solutions): evolves each of the constructed solutions in the initiation
    phase, running an incremental best first search over each solution, using mono_objective_score as goodness metric and sorting
    candidates of each solution by score.

    local_search_bitwise_neighborhood_sorted_score_r(initiated_solutions): evolves each of the constructed solutions in the initiation
    phase, running an incremental best first search over each solution, using mono_objective_score as goodness metric and sorting
    candidates of each solution by score. If each candidate does not improve the solution, it tries to remove each one of the other selected
    candidates and compares the metric, and selects the solution that improves the metric the most.

    local_search_bitwise_neighborhood_sorted_domination(initiated_solutions): evolves each of the constructed solutions in the initiation
    phase, running an incremental best first search over each solution, using domination as goodness metric and sorting
    candidates of each solution by score.

    update_nds(solutions): at the end of each GRASP iteration, the global self.nds list of solutions is updated based on the
        constructed and evolved solutions in such iteration.

    """

    def __init__(self, dataset:str="test", iterations:int=20, solutions_per_iteration:int=10, max_evaluations:int=0, init_type:str="stochastically",
                 local_search_type:str="best_first_neighbor_random", path_relinking_mode:str="None", seed:int=None, debug_mode:bool=False,
                 tackle_dependencies:bool=False):
        """
        :param dataset: integer number: 1 or 2
        :param iterations: integer (default 20), number of GRASP construct+local_search repetitions
        :param solutions_per_iteration: number of GRASPSolutions to be initially created and evolved in local search
        :param init_type: type of initialization
        :param local_search_type: type of search to evolve initiated solutions
        :param seed: int. seed for random generation of solutions in the first phase of each GRASP iteration
        """

        super().__init__(dataset, seed, debug_mode, tackle_dependencies)

        #self.dataset = Dataset(dataset)
        #self.dataset_name = dataset

        self.iterations:int = iterations
        self.solutions_per_iteration:int = solutions_per_iteration
        self.max_evaluations:int = max_evaluations

        self.nds = []
        self.num_evaluations:int = 0
        self.num_iterations:int = 0
        self.start:int = 0

        self.init_type:str = init_type
        self.local_search_type:str = local_search_type
        self.path_relinking_mode:str = path_relinking_mode

        #self.debug_mode = debug_mode
        #self.tackle_dependencies = tackle_dependencies

        #if seed is not None:
        #    np.random.seed(seed)

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

        self.executer = GRASPExecuter(algorithm=self)
        self.file:str = self.__class__.__name__+"-"+(str(dataset)+"-"+str(seed)+"-"+str(iterations)+"-"+str(solutions_per_iteration)
                                                 + "-"+str(init_type) + "-"+str(local_search_type) + "-"+str(path_relinking_mode)+".txt")
        # + "-"+str(max_evaluations) TODO

    def get_name(self):
        init = "stochastic" if self.init_type == "stochastically" else self.init_type
        local = "+"+self.local_search_type.replace(
            'best_first_neighbor_', '') if self.local_search_type != "None" else ""
        PR = "+PR" if self.path_relinking_mode != "None" else ""
        return "GRASP+"+str(self.iterations)+"+"+str(self.solutions_per_iteration)+"+"+str(self.max_evaluations)+"+"+init+local+PR

    def reset(self):
        self.nds = []
        self.num_evaluations = 0
        self.num_iterations = 0
        self.start = 0

    def stop_criterion(self, num_iterations, num_evaluations):
        if self.max_evaluations == 0:
            return num_iterations >= self.iterations
        else:
            return num_evaluations >= self.max_evaluations

    def add_evaluation(self, initiated_solutions):
        self.num_evaluations += 1
        # if(self.num_evaluations >= self.max_evaluations):
        if self.stop_criterion(self.num_iterations, self.num_evaluations):
            # self.update_nds(initiated_solutions)
            get_nondominated_solutions(initiated_solutions, self.nds)
            raise EvaluationLimit

    def run(self):
        """
        Core code of GRASP: initiation + local search + NDS update, repeated self.iterations times.
        :return (selected_list, seconds) list of ndarray and double.
                shape of selected is (len(self.nds), GraspSolution.dataset.num_pbis)
                    position ij==0 if solution i does not select candidate j
                    position ij==1 if solution i selects candidate j
                seconds is the time in seconds used to run all the GRASP iterations

        """
        self.reset()
        paretos = []
        self.start = time.time()

        # for _ in np.arange(self.iterations):
        self.num_iterations = 0
        try:
            while (not self.stop_criterion(self.num_iterations, self.num_evaluations)):
                # construction phase
                initiated_solutions = self.initialize()

                # local search phase
                if self.local_search != "None":
                    initiated_solutions = self.local_search(
                        initiated_solutions)

                if self.path_relinking_mode == "after_local":
                    initiated_solutions = self.path_relinking(
                        initiated_solutions)

                # repair population if dependencies tackled:
                if(self.tackle_dependencies):
                    initiated_solutions = self.repair_population_dependencies(
                        initiated_solutions)

                # update NDS with solutions constructed and evolved in this iteration
                # self.update_nds(initiated_solutions)
                get_nondominated_solutions(initiated_solutions, self.nds)

                self.num_iterations += 1

                if self.debug_mode:
                    paretos.append(self.nds)

        except EvaluationLimit:
            pass

        # return self.return_values()

        seconds = time.time() - self.start
        print("\nNDS created has", self.nds.__len__(), "solution(s)")
        #selected_list = []
        # for sol in self.nds:
        #    #   print("\n-----------------------")
        #    #   print(sol)
        #    selected_list.append(sol.selected)
        # return selected_list, seconds
        #genes,_ = generate_dataset_genes(self.dataset.id)


        #self.nds = format_population(self.nds, self.dataset)
        return {
            "population": self.nds,
            "time": seconds,
            "numGenerations": self.num_iterations,
            "numEvaluations": self.num_evaluations,
            "paretos": paretos
        }
        # return _results_in_victor_format(selected_list, seconds, self.num_iterations, genes, self.num_evaluations,paretos)

    # def return_values(self):
    #    seconds = time.time() - self.start
    #    print("\nNDS created has", self.nds.__len__(), "solution(s)")
    #    selected_list = []
    #    for sol in self.nds:
    #        #   print("\n-----------------------")
    #        #   print(sol)
    #        selected_list.append(sol.selected)
    #    # return selected_list, seconds
    #
    #    genes,_ = generate_dataset_genes(self.dataset.id)
    #    return _results_in_victor_format(selected_list, seconds, self.iterations, genes, self.num_evaluations)

    def repair_population_dependencies(self, solutions):
        for sol in solutions:
            sol.correct_dependencies(self.dataset)
        return solutions

    def init_solutions_stochastically(self):
        """
        candidates (pbis) are selected stochastically based on a rankin of the score of each pbi
        the ranking is scaled with values that sum up to 1. Each value is used as the probability to be chosen.
        :return solutions: list of GraspSolution
        """
        # scale candidates score, sum of scaled values is 1.0
        candidates_score_scaled = self.dataset.pbis_score / self.dataset.pbis_score.sum()

        # create GraspSolutions

        solutions = []
        for i in np.arange(self.solutions_per_iteration):
            sol = Solution(self.dataset, candidates_score_scaled)
            # avoid solution with 0 cost due to 0 candidates selected
            if np.count_nonzero(sol.selected) > 0:
                solutions.append(sol)
                i -= 1  # TODO ESTO NO DEBERIA SER EN EL ELSE?
        return solutions

    def init_solutions_uniform(self):
        """
        candidates (pbis) are selected uniformly 
        :return solutions: list of GraspSolution
        """
        # scale candidates score, sum of scaled values is 1.0
        candidates_score_scaled = np.full(
            self.dataset.pbis_score.size, 1/self.dataset.pbis_score.size)

        # create GraspSolutions
        solutions = []
        for i in np.arange(self.solutions_per_iteration):
            sol = Solution(self.dataset, candidates_score_scaled)
            # avoid solution with 0 cost due to 0 candidates selected
            if np.count_nonzero(sol.selected) > 0:
                solutions.append(sol)
                i -= 1
        return solutions

    def local_search_bitwise_neighborhood_sorted_score(self, initiated_solutions):
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
                # c1 = sol.total_cost
                # s1 = sol.total_satisfaction
                mo1 = sol.mono_objective_score
                # compute new cost, satisfaction and mo_score flipping candidate i in sol
                (c2, s2, mo2) = sol.try_flip(i, self.dataset.pbis_cost_scaled[i],
                                             self.dataset.pbis_satisfaction_scaled[i])
                self.add_evaluation(initiated_solutions)

                # no uso la dominancia porque cambiando solo 1 bit nunca se crea una solución dominante!
                # if c2 < c1 and s2 > s1:  # if neighbor  dominates, then overwrite former solution with neighbor

                # if neighbor has greater mono_objective_score, then overwrite former solution with neighbor
                if mo2 > mo1 and np.count_nonzero(
                        sol.selected) > 0:  # avoid solution with 0 cost due to 0 candidates selected
                    sol.flip(
                        i, self.dataset.pbis_cost_scaled[i], self.dataset.pbis_satisfaction_scaled[i])
                    self.add_evaluation(initiated_solutions)

        return initiated_solutions

    def local_search_bitwise_neighborhood_sorted_score_r(self, initiated_solutions):
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
                # c1 = sol.total_cost
                # s1 = sol.total_satisfaction
                mo1 = sol.mono_objective_score
                # compute new cost, satisfaction and mo_score flipping candidate i in sol
                (c2, s2, mo2) = sol.try_flip(i, self.dataset.pbis_cost_scaled[i],
                                             self.dataset.pbis_satisfaction_scaled[i])
                self.add_evaluation(initiated_solutions)
                # no uso la dominancia porque cambiando solo 1 bit nunca se crea una solución dominante!
                # if c2 < c1 and s2 > s1:  # if neighbor  dominates, then overwrite former solution with neighbor

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

    def local_search_bitwise_neighborhood_random(self, initiated_solutions):
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
                # c1 = sol.total_cost
                # s1 = sol.total_satisfaction
                mo1 = sol.mono_objective_score
                # compute new cost, satisfaction and mo_score flipping candidate i in sol
                (c2, s2, mo2) = sol.try_flip(i, self.dataset.pbis_cost_scaled[i],
                                             self.dataset.pbis_satisfaction_scaled[i])
                self.add_evaluation(initiated_solutions)
                # no uso la dominancia porque cambiando solo 1 bit nunca se crea una solución dominante!
                # if c2 < c1 and s2 > s1:  # if neighbor  dominates, then overwrite former solution with neighbor

                # if neighbor has greater mono_objective_score, then overwrite former solution with neighbor
                if mo2 > mo1 and np.count_nonzero(
                        sol.selected) > 0:  # avoid solution with 0 cost due to 0 candidates selected
                    sol.flip(
                        i, self.dataset.pbis_cost_scaled[i], self.dataset.pbis_satisfaction_scaled[i])
                    self.add_evaluation(initiated_solutions)

        return initiated_solutions

    def local_search_bitwise_neighborhood_random_domination(self, initiated_solutions):
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
                # c1 = sol.total_cost
                # s1 = sol.total_satisfaction
                # compute new cost, satisfaction and mo_score flipping candidate i in sol
                (c2, s2, mo2) = sol.try_flip(i, self.dataset.pbis_cost_scaled[i],
                                             self.dataset.pbis_satisfaction_scaled[i])
                self.add_evaluation(initiated_solutions)
                # no uso la dominancia porque cambiando solo 1 bit nunca se crea una solución dominante!
                # if c2 < c1 and s2 > s1:  # if neighbor  dominates, then overwrite former solution with neighbor

                # if neighbor has greater mono_objective_score, then overwrite former solution with neighbor
                if sol.is_dominated_by_value(c2, s2) and np.count_nonzero(
                        sol.selected) > 0:  # avoid solution with 0 cost due to 0 candidates selected
                    sol.flip(
                        i, self.dataset.pbis_cost_scaled[i], self.dataset.pbis_satisfaction_scaled[i])
                    self.add_evaluation(initiated_solutions)

        return initiated_solutions

    def local_search_bitwise_neighborhood_sorted_domination(self, initiated_solutions):
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
                # c1 = sol.total_cost
                # s1 = sol.total_satisfaction
                # compute new cost, satisfaction and mo_score flipping candidate i in sol
                (c2, s2, mo2) = sol.try_flip(i, self.dataset.pbis_cost_scaled[i],
                                             self.dataset.pbis_satisfaction_scaled[i])
                self.add_evaluation(initiated_solutions)
                # no uso la dominancia porque cambiando solo 1 bit nunca se crea una solución dominante!
                # if c2 < c1 and s2 > s1:  # if neighbor  dominates, then overwrite former solution with neighbor

                # if neighbor has greater mono_objective_score, then overwrite former solution with neighbor
                if sol.is_dominated_by_value(c2, s2) and np.count_nonzero(
                        sol.selected) > 0:  # avoid solution with 0 cost due to 0 candidates selected
                    sol.flip(
                        i, self.dataset.pbis_cost_scaled[i], self.dataset.pbis_satisfaction_scaled[i])
                    self.add_evaluation(initiated_solutions)

        return initiated_solutions

    def path_relinking(self, solutions):
        new_sols = []
        if len(self.nds) > 0:
            for solution in solutions:
                new_sols_path = []
                # get random solution from non dominated set
                random_nds_solution = random.choice(self.nds)
                # print("random",random_nds_solution.selected)
                # print("sol",solution.selected)
                # calculate distance from solution to goal random solution
                init_sol = copy.deepcopy(solution)
                distance = np.count_nonzero(
                    solution.selected != random_nds_solution.selected)
                # while distance greater than 0
                while(distance > 0):
                    # print("------------")
                    # print("dist",distance)
                    mono_score = solution.compute_mono_objective_score()
                    # calculate indexes of different bits
                    diff_bits = np.where(solution.selected !=
                                         random_nds_solution.selected)
                    diff_bits = diff_bits[0]
                    # random shuffle bits to flip
                    np.random.shuffle(diff_bits)
                    # print("diffs",diff_bits)
                    # for each different bit, try flip and store the best if improves monoscore
                    #best_mo = mono_score
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
                        # selected_flip = np.random.randint(
                        #    0, self.dataset.pbis_cost_scaled.size)

                    # print("selected",selected_flip)
                    # print("mono-best",mono_score,best_mo)

                    # flip selected bit (best by monobjective or random) of the solution (copy created to not replace former)
                    init_sol.flip(
                        selected_flip, self.dataset.pbis_cost_scaled[selected_flip], self.dataset.pbis_satisfaction_scaled[selected_flip])
                    self.add_evaluation(solutions+new_sols)
                    distance = distance - 1

                    # save intermediate solution of the path
                    new_sols_path.append(init_sol)
                    # print("sol",solution.selected)

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


# def _results_in_victor_format(nds, seconds, num_iterations, genes, num_evaluations,paretos):
#    """
#    esto será borrado cuando se programe un Evaluate que reciba el NDS (lista de listas de pbis) y el dataset usado,
#    para calcular las métricas fuera del algoritmo.
#    :param nds:
#    :param seconds:
#    :param num_iterations:
#    :param genes:
#    :param num_evaluations:
#    :return:
#    """
#    # convertir solución al formato de las soluciones de Victor
#    problem = Problem(genes, ["MAX", "MIN"])
#    final_nds_formatted = []
#    for solution in nds:
#        individual = Solution(problem.genes, problem.objectives, DEPENDENCIES)
#        for b in np.arange(len(individual.genes)):
#            individual.genes[b].included = solution[b]
#        individual.evaluate_fitness()
#        final_nds_formatted.append(individual)
#    print(seconds)
#    return {
#        "population": final_nds_formatted,
#        "time": seconds,
#        "numGenerations": num_iterations,
#        "numEvaluations": num_evaluations,
#        "paretos": paretos
#    }


def _get_options(argv=None):
    """out of class method used to gather dataset options
        :param argv: string with list of args and values
        :return: all the options
    """

    dataset = "1"
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
            dataset = str(arg)
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
    d, it, sols_per_it, init, search, s = _get_options(sys.argv[1:])
    g = GRASP(dataset=d, iterations=it, solutions_per_iteration=sols_per_it,
              init_type=init, local_search_type=search, seed=s)
    print("dataset:", d, "\niterations:", it, "\nsolutions_per_iteration:", sols_per_it, "\ninitialize_type:",
          init, "\nlocal_search_type:", search, "\nseed:", s)

    results = g.run()
    for y in results:
        print(y, ':', results[y])
