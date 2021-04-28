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
    set(i)
        sets self.selected[i]=1, and updates the solution cost, satisfaction and mono_objective_score
    unset(i)
        sets self.selected[i]=0, and updates the solution cost, satisfaction and mono_objective_score
    compute_mono_objective_score()
        a simple way to mix both satisfaction and cost metrics into a single one
    """

    def __init__(self, probabilities, costs, values):
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
        num_candidates = len(probabilities)
        self.selected = np.zeros(num_candidates)
        # samples a random number of candidates. prob of each candidate to be chosen in received in probabilites
        sampled = np.random.choice(np.arange(num_candidates), size=np.random.randint(num_candidates),
                                   replace=False, p=probabilities)
        self.selected[sampled] = 1

        self.total_cost = costs[sampled].sum()
        self.total_satisfaction = values[sampled].sum()

        self.mono_objective_score = self.compute_mono_objective_score()

    def compute_mono_objective_score(self):
        """
         computes self.mono_objective_score
         It does not overwrite self.mono_objective_score. That should be done by the user if desired.
        :return: mixture of satisfactions and costs of all selected candidates
        """
        return self.total_satisfaction / (self.total_cost + 1/len(np.where(self.selected == 1)))

    def set(self, i, i_cost, i_value):
        """
        :param i: new candidate selected
        :param i_cost: cost of such candidate
        :param i_value: value of such candidate
        """
        self.selected[i] = 1
        self.total_cost += i_cost
        self.total_satisfaction += i_value
        self.mono_objective_score = self.compute_mono_objective_score()

    def unset(self, i, i_cost, i_value):
        """
        :param i: candidate to be unselected
        :param i_cost: cost of such candidate
        :param i_value: value of such candidate
        """
        self.selected[i] = 0
        self.total_cost -= i_cost
        self.total_satisfaction -= i_value
        self.mono_objective_score = self.compute_mono_objective_score()

    def __str__(self):
        string = "PBIs selected in this Solution: "
        string += ' '.join(map(str, np.where(self.selected == 1)))
        string += "\nSatisfaction: " + str(self.total_satisfaction)
        string += "\nCost: " + str(self.total_cost)
        string += "\nMono Objective Score: " + str(self.mono_objective_score)
        return string
