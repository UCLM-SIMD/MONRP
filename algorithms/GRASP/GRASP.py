from algorithms.GRASP.Dataset import Dataset
import numpy as np

from algorithms.GRASP.GraspSolution import GraspSolution


class GRASP:
    """
    A GRASP search with two phases:
      - create initial set of solutions, based on a ranking of probabilities computed from
          pbi scores (as given by self.dataset.pbis_score)
      - local search, to improve solutions based on their neighbourhood

    Attributes
    ----------
    dataset: type algorithms.GRASP.Dataset
        each pbi in Dataset is a candidate to be included in one of the GRASP solutions.
        Dataset provides cost, satisfaction and score of each pbi.

    number_of_solutions: integer
        number of GRASPSolutions to be initially created and evolved in local search

    solutions: list of algorithms.GRASP.GraspSolution, len(solutions)=number_of_solutions
        it contains the current set of solutions

    Methods
    ---------
    run(): the method which starts the search.
     It calls the two phases of GRASP: initiate solutions and evolve them with local search

    init_solutions(): stochastically construct a former set of solutions,
     each of them containing a set of chosen candidates (pbis)

    """
    def __init__(self, dataset=1, number_of_solutions=10):
        """
        :param dataset: integer number: 1 or 2
        :param number_of_solutions: number of GRASPSolutions to be initially created and evolved in local search
        """
        self.dataset = Dataset(dataset)
        self.number_of_solutions = number_of_solutions
        self.solutions = []

    def run(self):
        # construct solution
        self.init_solutions()

        # local search

    def init_solutions(self):
        """
        candidates (pbis) are selected stochastically based on a rankin of the score of each pbi
        the ranking is scaled with values that sum up to 1. Each value is used as the probability to be chosen.
        :return:
        """
        # scale candidates score, sum of scaled values is 1.0
        candidates_score_scaled = self.dataset.pbis_score / self.dataset.pbis_score.sum()

        # create GraspSolutions
        for _ in np.arange(self.number_of_solutions):
            sol = GraspSolution(candidates_score_scaled, costs=self.dataset.pbis_cost_scaled, values=self.dataset.pbis_satisfaction_scaled)
            self.solutions.append(sol)


if __name__ == "__main__":
    g = GRASP(dataset=1, number_of_solutions=5)
    g.run()
