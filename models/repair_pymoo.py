from pymoo.core.repair import Repair

from models.problem_monrp import MONRProblem


class RepairPymoo(Repair):

    def _do(self, problem: MONRProblem, X,  **kwargs):

        deps = problem.dependencies
        for i in range(len(deps)):
            if deps[i] is not None:
                i_set = X[:, i]  # find where att i is set.
                for j in deps[i]:
                    X[i_set, j] = True # make sure xi --> xj


        return X