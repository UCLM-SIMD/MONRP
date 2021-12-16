from typing import List, Tuple
import numpy as np

from datasets.Dataset import Dataset


class Solution:
    """
    It represents a solution handled by algorithm search.
    """

    def __init__(self, dataset: Dataset, probabilities, selected=None, uniform=False):
        """
        """
        self.dataset: Dataset = dataset
        costs = dataset.pbis_cost_scaled
        values = dataset.pbis_satisfaction_scaled
        if uniform:
            genes = np.random.choice(2, len(costs))
            self.selected = np.array(genes, dtype=int)
            indexes = np.array(self.selected).nonzero()
            self.total_cost = costs[indexes].sum()
            self.total_satisfaction = values[indexes].sum()
        elif selected is not None:
            self.selected = np.array(selected, dtype=int)
            indexes = np.array(self.selected).nonzero()
            self.total_cost = costs[indexes].sum()
            self.total_satisfaction = values[indexes].sum()
        else:
            num_candidates = len(probabilities)
            self.selected = np.zeros(num_candidates, dtype=int)
            # samples a random number of candidates. prob of each candidate to be chosen in received in probabilities
            sampled = np.random.choice(np.arange(num_candidates), size=np.random.randint(num_candidates),
                                       replace=False, p=probabilities)
            self.selected[sampled] = 1

            self.total_cost = costs[sampled].sum()
            self.total_satisfaction = values[sampled].sum()

        self.mono_objective_score = self.compute_mono_objective_score()

    def compute_mono_objective_score(self) -> float:
        """
         computes self.mono_objective_score
         It does not overwrite self.mono_objective_score. That should be done by the user if desired (set and unset
         methods do overwrite it)
        :return: mixture of satisfactions and costs of all selected candidates
        """
        self.mono_objective_score = self.total_satisfaction / \
            (self.total_cost + 1 / len(np.where(self.selected == 1)))

        return self.mono_objective_score

    def evaluate(self) -> float:
        """Recalculates total cost and satisfaction counting selected requirements.
        """
        sel = (self.selected == 1)
        self.total_cost = self.dataset.pbis_cost_scaled[sel].sum()
        self.total_satisfaction = self.dataset.pbis_satisfaction_scaled[sel].sum(
        )
        return self.compute_mono_objective_score()

    def flip(self, i: int, i_cost: float, i_value: float) -> None:
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

    def try_flip(self, i: int, i_cost: float, i_value: float) -> float:
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

    def dominates(self, solution: "Solution") -> bool:
        """Return True if self dominates solution, in terms of cost and satisfaction
        """
        dominates = (self.total_cost < solution.total_cost) and (
            self.total_satisfaction > solution.total_satisfaction)

        dominates = dominates or (
            self.total_cost == solution.total_cost and self.total_satisfaction > solution.total_satisfaction)

        dominates = dominates or (
            self.total_cost < solution.total_cost and self.total_satisfaction == solution.total_satisfaction)

        return dominates

    def is_dominated_by_value(self, cost: float, satisfaction: float) -> bool:
        """Return True if self dominates solution, in terms of cost and satisfaction
        """
        dominated = (self.total_cost > cost) and (
            self.total_satisfaction < satisfaction)

        dominated = dominated or (
            self.total_cost == cost and self.total_satisfaction < satisfaction)

        dominated = dominated or (
            self.total_cost > cost and self.total_satisfaction == satisfaction)

        return dominated

    def dominates_all_in(self, solutions: List["Solution"]) -> bool:
        """Return True if self dominates all GraspSolution in solutions
        """
        for sol in solutions:
            if not self.dominates(sol):
                return False
        return True

    def is_dominated_by_any_in(self, solutions: List["Solution"]) -> bool:
        """Return True if self is dominated by any GraspSolution in solutions
        """
        for sol in solutions:
            if sol.dominates(self):
                return True
        return False

    def set_bit(self, index: int, value: int) -> None:
        """Sets a bit of a solution given its index to the value given, updating cost and satisfaction.
        """
        self.selected[index] = value
        i_cost = self.dataset.pbis_cost_scaled[index]
        i_value = self.dataset.pbis_satisfaction_scaled[index]
        mult = 1 if value == 1 else -1
        self.total_cost += i_cost*mult
        self.total_satisfaction += i_value*mult

    def correct_dependencies(self) -> None:
        """For each requirement selected, sets to 1 all requirements included in its dependencies.
        """
        # for each included gene
        for gene_index in range(len(self.selected)):
            if(self.selected[gene_index] == 1):
                # if has dependencies -> include all genes
                if self.dataset.dependencies[gene_index] is None:
                    continue
                for other_gene in self.dataset.dependencies[gene_index]:
                    #self.selected[other_gene-1] = 1
                    self.set_bit((other_gene), 1)

    def get_max_cost_satisfactions(self) -> float:
        return np.sum(self.dataset.pbis_cost_scaled), np.sum(self.dataset.pbis_satisfaction_scaled)

    def get_min_cost_satisfactions(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def __str__(self) -> str:
        string = "PBIs selected in this Solution: "
        string += ' '.join(map(str, np.where(self.selected == 1)))
        string += "\nSatisfaction: " + str(self.total_satisfaction)
        string += "\nCost: " + str(self.total_cost)
        string += "\nMono Objective Score: " + str(self.mono_objective_score)
        return string

    def print_genes(self) -> str:
        string = ""
        for gen in self.selected:
            string += str(gen)
        return string
