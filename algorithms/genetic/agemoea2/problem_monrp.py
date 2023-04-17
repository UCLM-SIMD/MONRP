import numpy as np
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.operators.sampling.rnd import BinaryRandomSampling


class MONRProblem(ElementwiseProblem):

    def __init__(self, num_pbis, costs, satisfactions, dependencies, num_deps):
        super().__init__(n_var=num_pbis, n_obj=2, n_constr=0, xl=0, xu=1, n_ieq_constr=num_deps, n_eq_constr=0)
        self.costs = costs
        self.satisfactions = satisfactions
        self.dependencies = dependencies

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] = []
        out["G"] = []

        #evaluate individual
        c = self.costs[x].sum()
        s = self.satisfactions[x].sum()
        s = 1 - s # f functions are to be minimized. so s has to be reverted
        out["F"] = out["F"] + [[c,s]]

        selected = [1 if tag else 0 for tag in x]
        #indicate constraint.
        # ej: r0-->r2 is x0 - x2 <=0 (if x0==1, x2 must be 1. else x2 may be 0 or 1).
        # note that number of constraints stacked in G must be equals to super().n_ieq_constr parameter
        for i in range(len(self.dependencies)):
            if self.dependencies[i] is not None:
                for j in self.dependencies[i]: #i-->j
                    out["G"] = out["G"] + [selected[i] - selected[j]]
        #if any value is G is > 0, then the solution is not feasible







