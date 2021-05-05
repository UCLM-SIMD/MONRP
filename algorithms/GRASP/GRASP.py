import time


from algorithms.GRASP.Dataset import Dataset
import numpy as np
import getopt
import sys

from algorithms.GRASP.GraspSolution import GraspSolution
from algorithms.genetic_nds.genetic_nds_utils import GeneticNDSUtils
from datasets import dataset1, dataset2
from models.individual import Individual
from models.problem import Problem


class GRASP:
    """
    __author__      = "Pablo.Bermejo@uclm.es"

    A GRASP search with two phases:
      - create initial set of solutions, based on a ranking of probabilities computed from
          pbi scores (as given by self.dataset.pbis_score)
      - local search, to improve solutions based on their neighbourhood

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
                 local_search_type="best_first_neighbor", seed=None):
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
        if seed is not None:
            np.random.seed(seed)

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

        genes = dataset1.generate_dataset1_genes() if self.dataset.id == 1 else dataset2.generate_dataset2_genes()

        return _results_in_victor_format(selected_list, seconds, self.iterations, genes)

    def init_solutions(self):
        """
        candidates (pbis) are selected stochastically based on a rankin of the score of each pbi
        the ranking is scaled with values that sum up to 1. Each value is used as the probability to be chosen.
        :return solutions: list of GraspSolution
        """
        # scale candidates score, sum of scaled values is 1.0
        candidates_score_scaled = self.dataset.pbis_score / self.dataset.pbis_score.sum()

        # create GraspSolutions

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
    d, it, sols_per_it, search, s = _get_options(sys.argv[1:])
    g = GRASP(dataset=d, iterations=it, solutions_per_iteration=sols_per_it, local_search_type=search, seed=s)
    print("dataset:", d, "\niterations:", it, "\nsolutions_per_iteration:", sols_per_it, "\nlocal_search_type:",
          search, "\nseed:", s)

    results = g.run()
    for y in results:
        print(y, ':', results[y])
