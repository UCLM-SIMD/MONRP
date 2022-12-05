from models.Solution import Solution
import evaluation.metrics as metrics

"""
   perform solution subset selection on received set of solutions
   """
def search_solution_subset(sss_type: int, subset_size: int, solutions: [Solution]) -> [Solution]:
    if sss_type == 0:
        subset = greedy_hv_sss(subset_size, solutions)

    return subset


""" finds the subset which maximizes HV with subset_size of solutions.
    The search is a basic/tradicional greedy forward search based on the HV metric
    A fixed reference point is assumed during the search (always the same), as in 
    'Greedy Hypervolume Subset Selection in Low Dimensions,Evolutionary Computation 24(3): 521-544'
    where fixed r is upper bounds (in minimization problems), that is, the nadir point

    """
def greedy_hv_sss(subset_size: int, solutions: [Solution]) -> [Solution]:
    if len(solutions) < subset_size:
        print('|solutions| < subset_size parameter!! Solution subset set to original final solution');
        # warnings.warn('|solutions| < subset_size parameter!! Solution subset set to original final solution', UserWarning)
        return solutions

    indices_selected = []
    subset = []
    # metrics.calculate_hypervolume(solutions, ref_x=1.1, ref_y=1.1) #for plotting whold nds before subset selection
    for _ in range(0, subset_size):
        best_hv = -1
        best_index = -1
        for i in range(0, len(solutions)):
            if not i in indices_selected:
                subset.insert(len(subset), solutions[i])
                hv = metrics.calculate_hypervolume(subset, ref_x=1.1, ref_y=1.1)
                if hv > best_hv:
                    best_hv = hv
                    best_index = i
                del subset[-1]
        if best_index != -1:
            subset.insert(len(subset), solutions[best_index])
            indices_selected.insert(len(indices_selected), best_index)
    # plot_solutions(subset)
    return subset