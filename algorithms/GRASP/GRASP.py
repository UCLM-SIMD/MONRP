from algorithms.abstract_default.algorithm import Algorithm
from algorithms.GRASP.grasp_executer import GRASPExecuter
import time

from algorithms.GRASP.Dataset import Dataset
import numpy as np
import getopt
import sys

from algorithms.GRASP.GraspSolution import GraspSolution
from datasets import dataset1, dataset2
from models.solution import Solution
from models.problem import Problem


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
        possible values are: best_first_neighbor (default), best_first_neighbor_random, best_first_neighbor_sorted_score

    Methods
    ---------
    get_name(): returns name of the algorithm.

    run(): the method which starts the search.
     It calls the two phases of GRASP: initiate solutions and evolve them with local search

    init_solutions_stochastically(): stochastically construct a former set of solutions,
     each of them containing a set of chosen candidates (pbis)

    init_solutions_uniform(): uniformly construct a former set of solutions,
     each of them containing a set of chosen candidates (pbis)

    local_search_bitwise_neighborhood(initiated_solutions): evolves each of the constructed solutions in the initiation
    phase, running an incremental best first search over each solution, using mono_objective_score as goodness metric 

    local_search_bitwise_neighborhood_random(initiated_solutions): evolves each of the constructed solutions in the initiation
    phase, running an incremental best first search over each solution, using mono_objective_score as goodness metric and randomness

    best_first_neighbor_sorted_score(initiated_solutions): evolves each of the constructed solutions in the initiation
    phase, running an incremental best first search over each solution, using mono_objective_score as goodness metric and sorting
    candidates of each solution by score.

    update_nds(solutions): at the end of each GRASP iteration, the global self.NDS list of solutions is updated based on the
        constructed and evolved solutions in such iteration.

    """

    def __init__(self, dataset="1", iterations=20, solutions_per_iteration=10, init_type="stochastically",
                 local_search_type="best_first_neighbor", seed=None):
        """
        :param dataset: integer number: 1 or 2
        :param iterations: integer (default 20), number of GRASP construct+local_search repetitions
        :param solutions_per_iteration: number of GRASPSolutions to be initially created and evolved in local search
        :param init_type: type of initialization
        :param local_search_type: type of search to evolve initiated solutions
        :param seed: int. seed for random generation of solutions in the first phase of each GRASP iteration
        """
        self.dataset = Dataset(dataset)
        self.dataset_name = dataset
        self.iterations = iterations
        self.solutions_per_iteration = solutions_per_iteration
        self.NDS = []
        self.init_type = init_type
        self.local_search_type = local_search_type
        if seed is not None:
            np.random.seed(seed)

        if self.init_type == "stochastically":
            self.initialize = self.init_solutions_stochastically
        elif self.init_type == "uniform":
            self.initialize = self.init_solutions_uniform

        if self.local_search_type == "best_first_neighbor":
            self.local_search = self.local_search_bitwise_neighborhood
        elif self.local_search_type == "best_first_neighbor_random":
            self.local_search = self.local_search_bitwise_neighborhood_random
        elif self.local_search_type == "best_first_neighbor_sorted_score":
            self.local_search = self.local_search_bitwise_neighborhood_sorted_score

        self.executer = GRASPExecuter(algorithm=self)
        self.file = self.__class__.__name__+"-"+(str(dataset)+"-"+str(seed)+"-"+str(iterations)+"-"+str(solutions_per_iteration)
                                                 + "-"+str(init_type) + "-"+str(local_search_type)+".txt")

    def get_name(self):
        return "GRASP "+self.init_type+" "+self.local_search_type

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
            initiated_solutions = self.initialize()

            # local search phase
            initiated_solutions = self.local_search(initiated_solutions)

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

        genes = dataset1.generate_dataset1_genes(
        ) if self.dataset.id == "1" else dataset2.generate_dataset2_genes()

        return _results_in_victor_format(selected_list, seconds, self.iterations, genes)

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
            sol = GraspSolution(candidates_score_scaled, costs=self.dataset.pbis_cost_scaled,
                                values=self.dataset.pbis_satisfaction_scaled)
            # avoid solution with 0 cost due to 0 candidates selected
            if np.count_nonzero(sol.selected) > 0:
                solutions.append(sol)
                i -= 1
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
            sol = GraspSolution(candidates_score_scaled, costs=self.dataset.pbis_cost_scaled,
                                values=self.dataset.pbis_satisfaction_scaled)
            # avoid solution with 0 cost due to 0 candidates selected
            if np.count_nonzero(sol.selected) > 0:
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
                    sol.flip(
                        i, self.dataset.pbis_cost_scaled[i], self.dataset.pbis_satisfaction_scaled[i])

        return initiated_solutions

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

                # no uso la dominancia porque cambiando solo 1 bit nunca se crea una solución dominante!
                # if c2 < c1 and s2 > s1:  # if neighbor  dominates, then overwrite former solution with neighbor

                # if neighbor has greater mono_objective_score, then overwrite former solution with neighbor
                if mo2 > mo1 and np.count_nonzero(
                        sol.selected) > 0:  # avoid solution with 0 cost due to 0 candidates selected
                    sol.flip(
                        i, self.dataset.pbis_cost_scaled[i], self.dataset.pbis_satisfaction_scaled[i])

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

                # no uso la dominancia porque cambiando solo 1 bit nunca se crea una solución dominante!
                # if c2 < c1 and s2 > s1:  # if neighbor  dominates, then overwrite former solution with neighbor

                # if neighbor has greater mono_objective_score, then overwrite former solution with neighbor
                if mo2 > mo1 and np.count_nonzero(
                        sol.selected) > 0:  # avoid solution with 0 cost due to 0 candidates selected
                    sol.flip(
                        i, self.dataset.pbis_cost_scaled[i], self.dataset.pbis_satisfaction_scaled[i])

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
                if np.array_equal(sol.selected, nds_sol.selected):
                    insert = False
                    break
                else:
                    if sol.dominates(nds_sol):
                        now_dominated.append(nds_sol)
                    # do not insert if sol is dominated by a solution in self.NDS
                    if nds_sol.dominates(sol):
                        insert = False
                        break

            # sol is inserted if it is not dominated by any solution in self.NDS,
            # then all solutions in self.NDS dominated by sol are removed
            if insert:
                self.NDS.append(sol)
                for dominated in now_dominated:
                    self.NDS.remove(dominated)


def _results_in_victor_format(nds, seconds, num_iterations, genes):
    """
    esto será borrado cuando se programe un Evaluate que reciba el NDS (lista de listas de pbis) y el dataset usado,
    para calcular las métricas fuera del algoritmo.
    :param nds:
    :param seconds:
    :param num_iterations:
    :param genes:
    :return:
    """
    # convertir solución al formato de las soluciones de Victor
    problem = Problem(genes, ["MAX", "MIN"])
    final_nds_formatted = []
    for solution in nds:
        individual = Solution(problem.genes, problem.objectives)
        for b in np.arange(len(individual.genes)):
            individual.genes[b].included = solution[b]
        individual.evaluate_fitness()
        final_nds_formatted.append(individual)

    return {
        "population": final_nds_formatted,
        "time": seconds,
        "numGenerations": num_iterations,
    }


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
