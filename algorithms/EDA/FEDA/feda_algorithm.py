from typing import Any, Dict, List
from algorithms.EDA.eda_algorithm import EDAAlgorithm
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from evaluation.get_nondominated_solutions import get_nondominated_solutions
from models.Solution import Solution
from algorithms.EDA.UMDA.umda_executer import UMDAExecuter
from collections import defaultdict

import time
import numpy as np


class FEDAAlgorithm(EDAAlgorithm):
    """ Fixed structure EDA
    _author__      = "Victor.PerezPiqueras@uclm.es" "Pablo.Bermejo@uclm.es"

    Given an initial set of requirements (pbis) dependencies in the form of
    X1-->X2, FEDA uses this knowledge as a prefixed structure.
    e.g: we can have an acyclic graph like this: G={0-->2, 1-->2, 3, 2-->4}, where pbis 0,1,3 do not have parents,
    parents(2)={0,1} and parents(4)={2}.

    Thus, learning is not structural and only applies to data.
    Sampling is always performed following a topological (ancestral) order ([3,0,1,2,4] in the example above).

    Algorithm is as follows:

    1. Sampling of First generation:
    -- If X does not have any parents, then sample using  P(X)=1/self.dataset.num_pbis
    -- If any Y in parents(X) is set to 1, then X=1, else use P(X)=1/self.dataset.num_pbis

    do{

    2. Learning
    -- If X does not have any parents, learn its marginal probability
    -- Conditional: P(X| each Y in parents(X)==0)
    In the example above, P(2| 0==0,1==0). thus,
    we only need to learn P(X|parents(X)) from individuals where all Y in parents(X) are == 0.
    This means that conditional probability can be stored in an unidimensional array (self.cond_probability_model).

    3. Sampling
    -- In the case of requirements without parents, use marginal probability
    -- In any Y in parents(X) is set to 1, then X=1, else use P(X|parents(X)==0)

    }while(!stop_criterion)
    """

    def __init__(self, dataset_name: str = "1", random_seed: int = None, debug_mode: bool = False,
                 tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0):

        super().__init__(dataset_name, random_seed, debug_mode, tackle_dependencies,
                         population_length, max_generations, max_evaluations)

        self.population = None
        # self.executer = UMDAExecuter(algorithm=self)

        self.file: str = (
            f"{str(self.__class__.__name__)}-{str(dataset_name)}-{str(random_seed)}-{str(population_length)}-"
            f"{str(max_generations)}-{str(max_evaluations)}.txt")

        # self.probability_model = np.full(self.dataset.num_pbis, 1/self.dataset.num_pbis)

        self.graph = defaultdict(list)  # self.graph[p]  returns list of children(p), if any.
        self.parents_of = defaultdict(list)
        self.topological_order = self.compute_topological_order()
        self.orphans = self.find_orphans()


    def get_name(self) -> str:
        return (f"FEDA{str(self.population_length)}+{str(self.max_generations)}+"
                f"{str(self.max_evaluations)}")

    def run(self) -> Dict[str, Any]:
        self.reset()
        paretos = []
        start = time.time()

        self.population = self.init_population()
        self.evaluate(self.population, self.best_individual)

        try:
            while not self.stop_criterion(self.num_generations, self.num_evaluations):
                # selection
                individuals = self.select_individuals(self.population)

                # learning
                # probability_model = self.learn_probability_model(individuals)

                # replacement
                # self.population = self.sample_new_population(probability_model)

                # repair population if dependencies tackled:
                # if (self.tackle_dependencies):
                #   self.population = self.repair_population_dependencies(
                #       self.population)

                # evaluation
                self.evaluate(self.population, self.best_individual)

                # update nds with solutions constructed and evolved in this iteration
                get_nondominated_solutions(self.population, self.nds)

                self.num_generations += 1

                if self.debug_mode:
                    paretos.append(self.nds)

        except EvaluationLimit:
            pass

        end = time.time()

        print("\nNDS created has", self.nds.__len__(), "solution(s)")

        return {"population": self.nds,
                "time": end - start,
                "numGenerations": self.num_generations,
                "best_individual": self.best_individual,
                "numEvaluations": self.num_evaluations,
                "paretos": paretos
                }

    '''
    1. Sampling of First generation:
        -- If X does not have any parents, then sample using  P(X)=1/self.dataset.num_pbis
        -- If any Y in parents(X) is set to 1, then X=1, else use P(X)=1/self.dataset.num_pbis
    '''

    def init_population(self) -> List[Solution]:

        population = []
        probs = np.full(self.dataset.num_pbis, 1/self.dataset.num_pbis)

        for _ in np.arange(self.population_length):
            sample_selected = np.full(self.dataset.num_pbis, 0)
            # sample whole individual using P(X)= 1/self.dataset.num_pbis for all X
            while 1 not in sample_selected:
                sample_selected = np.random.binomial(1, probs)
            solution = Solution(self.dataset, None, selected=sample_selected)

            # now follow topological order to check if any X must be set to 1
            for x in self.topological_order:
                for p in self.parents_of[x]:
                    if solution.selected[p] == 1:
                        solution.selected[x] = 1
            population.append(solution)

        return population






    '''
     Linear ordering of its vertices such that for every directed edge uv from vertex u to vertex v,
     u comes before v in the ordering
    Original code obtained from https://www.geeksforgeeks.org/python-program-for-topological-sorting/
    '''

    def compute_topological_order(self):

        v = self.dataset.num_pbis  # number of vertices (nodes)

        # create list of edges
        for parent in np.arange(v):
            sons = self.dataset.list_of_sons[parent]
            if sons is not None:
                for s in sons:
                    self.graph[parent].append(s)
                    self.parents_of[s].append(parent)

        print(self.graph)
        # Mark all the vertices as not visited
        visited = [False] * v
        order = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(v):
            if not visited[i]:
                self.topological_sort_util(i, visited, order)

        print("Topological order of graph structure is:\n", order)

        return order

    '''
    recursively called, first time from self.get_topological_order
    Original code obtained from https://www.geeksforgeeks.org/python-program-for-topological-sorting/
    '''


    def topological_sort_util(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if not visited[i]:
                self.topological_sort_util(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    def find_orphans(self) -> np.array:
        orphans = np.arange(self.dataset.num_pbis)
        remove_list = []

        for x in np.arange(self.dataset.num_pbis):
            if any([True for _, v in self.graph.items() if x in v]):
                remove_list.append(x)

        orphans = np.delete(orphans, remove_list)
        return orphans





if __name__ == "__main__":
    feda = FEDAAlgorithm(dataset_name="test")
    feda.run()
