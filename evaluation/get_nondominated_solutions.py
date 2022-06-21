from typing import List
import numpy as np

from algorithms.abstract_algorithm.abstract_algorithm import plot_solutions
from models.Solution import Solution


def get_nondominated_solutions(solutions: List[Solution], nds=None) -> List[Solution]:
    """Given a set of solutions and a set of NDS, this method updates the NDS, inserting nondominated
    solutions of the solutions set and removing those from the NDS that are then dominated by other solutions.
    """
    if nds is None:
        nds = []

    for sol in solutions:
        insert = True

        # find which solutions, if any, in self.NDS are dominated by sol
        # if sol is dominated by any solution in self.NDS, then search is stopped and sol is discarded
        now_dominated = []
        for nds_sol in nds:
            if hasattr(sol, 'selected') and np.array_equal(sol.selected, nds_sol.selected):
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
            nds.append(sol)
            for dominated in now_dominated:
                nds.remove(dominated)

   # plot_solutions(nds)
    return nds



