import numpy as np


class GraspSolution:
    """
    It represents a solution handled by GRASP search.

    Attributes
    ----------
    selected: numpy ndarray shape(number of candidates to be chosen)
        Initially all slots are set to 0 (not selected). When a slot is set to 1, that candidate (pbi, feature,etc...) is chosen.

    total_cost: integer or double
        sum of cost of all the selected candidates in selected. Metric to be minimized.

    total_satisfaction: double
        it represents the total importance of all candidates together. Metric to be maximized.

    mono_objective_score: double
        in case it is necessary, a single metric which is the mixture of cost and satisfaction


    Methods
    -------
    flip(i,cost_i, value_i)
        swaps value of self.selected[i] (0 to 1 or 1 to 0), and updates the solution cost, satisfaction and mono_objective_score
    compute_mono_objective_score()
        a simple way to mix both satisfaction and cost metrics into a single one
    try_flip(i,cost_i, value_i)
        it simulates the result of flip(i,cost_i,value_i) and returns the would-be new cost, value and mono_objective_score
    """

    def __init__(self, probabilities, costs, values,selected=None,uniform=False):
        """
        :param probabilities: numpy ndarray
            probabilities[i] is the probability in range [0-1] to set self.selected[i] to 1.
            len(probabilities)==len(selected)
        :param costs: numpy ndarray, shape is len(selected)
            costs[i] is the cost associated to candidate (e.g. pbi) i.
            when called from GRASP object, it is recommended to use scaled values such as self.dataset.pbis_cost_scaled
        :param values: numpy ndarray, shape is len(selected)
            values[i] is the goodness metric of candidate i.
            when called from GRASP object, it is recommended to use scaled values such as self.dataset.pbis_satisfaction_scaled
        """
        if uniform:
            genes = np.random.choice(2, len(costs))
            self.selected=genes
            indexes = np.array(self.selected).nonzero()
            self.total_cost = costs[indexes].sum()
            self.total_satisfaction = values[indexes].sum()
        elif selected is not None:
            self.selected=np.array(selected)
            indexes = np.array(self.selected).nonzero()
            self.total_cost = costs[indexes].sum()
            self.total_satisfaction = values[indexes].sum()
        else:
            num_candidates = len(probabilities)
            self.selected = np.zeros(num_candidates)
            # samples a random number of candidates. prob of each candidate to be chosen in received in probabilities
            sampled = np.random.choice(np.arange(num_candidates), size=np.random.randint(num_candidates),
                                    replace=False, p=probabilities)
            self.selected[sampled] = 1

            self.total_cost = costs[sampled].sum()
            self.total_satisfaction = values[sampled].sum()

        self.mono_objective_score = self.compute_mono_objective_score()

    def compute_mono_objective_score(self):
        """
         computes self.mono_objective_score
         It does not overwrite self.mono_objective_score. That should be done by the user if desired (set and unset
         methods do overwrite it)
        :return: mixture of satisfactions and costs of all selected candidates
        """
        return self.total_satisfaction / (self.total_cost + 1 / len(np.where(self.selected == 1)))

    def flip(self, i, i_cost, i_value):
        """
        :param i: new candidate to be (un)selected
        :param i_cost: cost of such candidate
        :param i_value: value of such candidate
        """
        if self.selected[i] == 0:
            self.selected[i] = 1
            self.total_cost += i_cost
            self.total_satisfaction += i_value

        elif self.selected[i] == 1:
            self.selected[i] = 0
            self.total_cost -= i_cost
            self.total_satisfaction -= i_value
        self.mono_objective_score = self.compute_mono_objective_score()

    def try_flip(self, i, i_cost, i_value):
        """
        This method simulates flip(i, i_cost, i_value), without changing any attribute from the object
        :param i: new candidate to be (un)selected
        :param i_cost: cost of such candidate
        :param i_value: value of such candidate
        :return the would-be new cost, value and mono_objective_score
        """
        if self.selected[i] == 0:
            new_cost = self.total_cost + i_cost
            new_satisfaction = self.total_satisfaction + i_value
            smooth = len(np.where(self.selected == 1)) + 1

        else:  # if self.selected[i] == 1:
            new_cost = self.total_cost - i_cost
            new_satisfaction = self.total_satisfaction - i_value
            smooth = len(np.where(self.selected == 1)) - 1
            smooth = 1 if smooth == 0 else smooth

        return (new_cost, new_satisfaction,
                new_satisfaction / (new_cost + 1 / smooth))

    def dominates(self, solution):
        """
        :param solution: GRASPSolution
        :return: True if self dominates solution, in terms of cost and satisfaction
        """
        dominates = (self.total_cost < solution.total_cost) and (
            self.total_satisfaction > solution.total_satisfaction)

        dominates = dominates or (
            self.total_cost == solution.total_cost and self.total_satisfaction > solution.total_satisfaction)

        dominates = dominates or (
            self.total_cost < solution.total_cost and self.total_satisfaction == solution.total_satisfaction)
        
        return dominates

    def is_dominated_by_value(self, cost, satisfaction):
        """
        :param cost: double
        :param satisfaction: double
        :return: True if self dominates solution, in terms of cost and satisfaction
        """
        dominated = (self.total_cost > cost) and (
            self.total_satisfaction < satisfaction)

        dominated = dominated or (
            self.total_cost == cost and self.total_satisfaction < satisfaction)

        dominated = dominated or (
            self.total_cost > cost and self.total_satisfaction == satisfaction)
        #print(self.total_cost,self.total_satisfaction,dominated,cost,satisfaction)
        return dominated

    def dominates_all_in(self, solutions):
        """
        :param solutions: list of GraspSolution
        :return: True if self dominates all GraspSolution in solutions
        """
        for sol in solutions:
            if not self.dominates(sol):
                return False
        return True

    def is_dominated_by_any_in(self, solutions):
        """

        :param solutions: list of GraspSolution
        :return: True if self is dominated by any GraspSolution in solutions
        """
        for sol in solutions:
            if sol.dominates(self):
                return True
        return False

    def __str__(self):
        string = "PBIs selected in this Solution: "
        string += ' '.join(map(str, np.where(self.selected == 1)))
        string += "\nSatisfaction: " + str(self.total_satisfaction)
        string += "\nCost: " + str(self.total_cost)
        string += "\nMono Objective Score: " + str(self.mono_objective_score)
        return string
