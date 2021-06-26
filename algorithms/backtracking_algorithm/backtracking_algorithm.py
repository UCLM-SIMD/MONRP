from models.solution import Solution
from datasets.dataset_gen_generator import generate_dataset_genes
from models.problem import Problem
from algorithms.backtracking_algorithm.Solution import Solution as BacktrackingSolution
from algorithms.GRASP.GraspSolution import GraspSolution
from algorithms.GRASP.Dataset import Dataset
from algorithms.abstract_default.algorithm import Algorithm
import numpy as np
import itertools


class BacktrackingAlgorithm(Algorithm):
    def __init__(self, dataset, seed=None):
        self.dataset = Dataset(dataset)
        self.dataset_name = dataset
        self.NDS = []
        if seed is not None:
            np.random.seed(seed)

    def run(self):
        num_candidates = len(self.dataset.pbis_cost_scaled)
        probs = np.zeros(num_candidates, dtype="int")
        solution = BacktrackingSolution(probs, costs=self.dataset.pbis_cost_scaled,
                            values=self.dataset.pbis_satisfaction_scaled)

        solutions = np.array(
            list(itertools.product([0, 1], repeat=num_candidates)))
        #print(solutions[0])
        solutions = np.delete(solutions, [0], axis=0)
        #print(solutions[0])
        counter = 0
        sol_set = []
        #print(len(solutions))
        np.random.shuffle(solutions)
        for sol in solutions:
            solution = BacktrackingSolution(sol, costs=self.dataset.pbis_cost_scaled,
                                values=self.dataset.pbis_satisfaction_scaled)
            # print(solution.selected)
            counter += 1
            sol_set.append(solution)
            if counter % 1000 == 0:
                #counter = 0
                self.update_nds(sol_set)
                sol_set = []
                # print("--")
                #print(counter, len(self.NDS), solution.selected,
                #      solution.total_cost, solution.total_satisfaction)
                #print(self.NDS[0].total_cost,
                #      self.NDS[0].total_satisfaction, self.NDS[0].selected)
            
        self.update_nds(sol_set)
        
        genes = generate_dataset_genes(self.dataset.id)
        problem = Problem(genes, ["MAX", "MIN"])
        final_nds_formatted = []
        for solution in self.NDS:
            #print(solution)
            individual = Solution(problem.genes, problem.objectives)
            for b in np.arange(len(individual.genes)):
                individual.genes[b].included = solution.selected[b]
            individual.evaluate_fitness()
            final_nds_formatted.append(individual)

        return {
            "population": final_nds_formatted,
        }

    def get_name(self):
        return "backtracking"

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

    def updateNDS(self, new_population):
        new_nds = []
        self.NDS.extend(new_population)
        for ind in self.NDS:
            dominated = False
            for other_ind in self.NDS:
                if other_ind.dominates(ind):
                    dominated = True
                    break
            if not dominated:
                new_nds.append(ind)
        new_nds = list(set(new_nds))
        self.NDS = new_nds
