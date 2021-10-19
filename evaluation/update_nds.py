import numpy as np
def get_nondominated_solutions(solutions,nds=[]):
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
        for nds_sol in nds:
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
            nds.append(sol)
            for dominated in now_dominated:
                nds.remove(dominated)

    return nds