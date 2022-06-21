import math
from decimal import Decimal, ROUND_HALF_EVEN
from typing import List, Tuple
import numpy as np

from datasets.Dataset import Dataset


class Solution:
    """
    It represents a solution handled by algorithm search.
    """

<<<<<<< HEAD
<<<<<<< HEAD
    def __init__(self, dataset, probabilities, selected=None, uniform=False, *, cost=None, satisfaction=None):
=======
    def __init__(self, dataset: Dataset, probabilities, selected=None, uniform=False, *, cost=None, satisfaction=None):
>>>>>>> 9617fc4f (extract_postMetrics.py computes and updates outputs .json with: gd+, unfr and reference pareto front.)
=======
    def __init__(self, dataset, probabilities, selected=None, uniform=False, *, cost=None, satisfaction=None):
>>>>>>> a1359f27 (solved issue when comparing new solutions to nds (.isclose). now solution subset search has a better general ref point.)
        """
        """
        if cost is None or satisfaction is None:
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
<<<<<<< HEAD

                self.selected = np.zeros(dataset.num_pbis, dtype=int)
                self.selected[selected] = 1
                # indexes = np.array(self.selected).nonzero()
                self.total_cost = costs[selected].sum()
                self.total_satisfaction = values[selected].sum()
=======
                self.selected = np.array(selected, dtype=int)
                indexes = np.array(self.selected).nonzero()
                self.total_cost = costs[indexes].sum()
                self.total_satisfaction = values[indexes].sum()
>>>>>>> 9617fc4f (extract_postMetrics.py computes and updates outputs .json with: gd+, unfr and reference pareto front.)
            else:
                num_candidates = len(probabilities)
                self.selected = np.zeros(num_candidates, dtype=int)
                # samples a random number of candidates. prob of each candidate to be chosen in received in probabilities
                sampled = np.random.choice(np.arange(num_candidates), size=np.random.randint(num_candidates),
                                           replace=False, p=probabilities)
                self.selected[sampled] = 1

                self.total_cost = costs[sampled].sum()
                self.total_satisfaction = values[sampled].sum()

<<<<<<< HEAD


=======
>>>>>>> 9617fc4f (extract_postMetrics.py computes and updates outputs .json with: gd+, unfr and reference pareto front.)
            self.mono_objective_score = self.compute_mono_objective_score()
        else: # this branch should be used only from ./extract_postMetrics.py scrip
            self.total_cost = cost
            self.total_satisfaction = satisfaction






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
            self.total_cost = self.total_cost + i_cost
            self.total_satisfaction =  self.total_satisfaction+  i_value
            self.total_satisfaction = 1 if self.total_satisfaction > 1 else self.total_satisfaction
            #by precision loss

        elif self.selected[i] == 1:
            self.selected[i] = 0
            self.total_cost = self.total_cost - i_cost
            self.total_satisfaction = self.total_satisfaction - i_value

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
    # read https://davidamos.dev/the-right-way-to-compare-floats-in-python/ for floats comparison
<<<<<<< HEAD
<<<<<<< HEAD
    # we chose 0.001 as abs_tol difference as margin to decide values are equals. otherwise,
=======
    # we chose 0.01 as abs_tol difference as margin to decide values are equals. otherwise,
>>>>>>> 9617fc4f (extract_postMetrics.py computes and updates outputs .json with: gd+, unfr and reference pareto front.)
=======
    # we chose 0.001 as abs_tol difference as margin to decide values are equals. otherwise,
>>>>>>> a1359f27 (solved issue when comparing new solutions to nds (.isclose). now solution subset search has a better general ref point.)
    # when plotting solutions they seem equal since the difference is too tiny. it might be
    #changed to 0.001 or even 0.0001.
    #by default, one solution equals to other dominates it.
    def dominates(self, solution: "Solution", equals_dominates = True) -> bool:
        """Return True if self dominates solution, in terms of cost and satisfaction
        """
        this_cost = Decimal(self.total_cost).quantize(Decimal('.12345'), rounding=ROUND_HALF_EVEN)
        other_cost = Decimal(solution.total_cost).quantize(Decimal('.12345'), rounding=ROUND_HALF_EVEN)
        this_satisfaction = Decimal(self.total_satisfaction).quantize(Decimal('.12345'), rounding=ROUND_HALF_EVEN)
        other_satisfaction = Decimal(solution.total_satisfaction).quantize(Decimal('.12345'), rounding=ROUND_HALF_EVEN)


        dominates = (this_cost < other_cost) and (this_satisfaction > other_satisfaction)

        dominates = dominates or (
<<<<<<< HEAD
<<<<<<< HEAD
            math.isclose(self.total_cost,solution.total_cost,abs_tol=0.0001) and
            this_satisfaction > other_satisfaction)

        dominates = dominates or (this_cost < other_cost and
                math.isclose(self.total_satisfaction,solution.total_satisfaction,abs_tol=0.0001))
=======
            math.isclose(self.total_cost,solution.total_cost,abs_tol=0.01) and
            this_satisfaction > other_satisfaction)

        dominates = dominates or (this_cost < other_cost and
                math.isclose(self.total_satisfaction,solution.total_satisfaction,abs_tol=0.01))
>>>>>>> 9617fc4f (extract_postMetrics.py computes and updates outputs .json with: gd+, unfr and reference pareto front.)
=======
            math.isclose(self.total_cost,solution.total_cost,abs_tol=0.0001) and
            this_satisfaction > other_satisfaction)

        dominates = dominates or (this_cost < other_cost and
                math.isclose(self.total_satisfaction,solution.total_satisfaction,abs_tol=0.0001))
>>>>>>> a1359f27 (solved issue when comparing new solutions to nds (.isclose). now solution subset search has a better general ref point.)

        #if both are equals, let one dominate the other and thus remove it
        if equals_dominates:
            dominates = dominates or (
<<<<<<< HEAD
<<<<<<< HEAD
                math.isclose(self.total_cost,solution.total_cost,abs_tol=0.0001) and
                math.isclose(self.total_satisfaction,solution.total_satisfaction,abs_tol=0.0001))
=======
                math.isclose(self.total_cost,solution.total_cost,abs_tol=0.01) and
                math.isclose(self.total_satisfaction,solution.total_satisfaction,abs_tol=0.01))
>>>>>>> 9617fc4f (extract_postMetrics.py computes and updates outputs .json with: gd+, unfr and reference pareto front.)
=======
                math.isclose(self.total_cost,solution.total_cost,abs_tol=0.0001) and
                math.isclose(self.total_satisfaction,solution.total_satisfaction,abs_tol=0.0001))
>>>>>>> a1359f27 (solved issue when comparing new solutions to nds (.isclose). now solution subset search has a better general ref point.)

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
                    if self.selected[other_gene]!=1:
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
