from typing import Any, Dict, List
from algorithms.EDA.eda_algorithm import EDAAlgorithm
<<<<<<< HEAD
from algorithms.abstract_algorithm.abstract_algorithm import plot_solutions
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from datasets import Dataset
from evaluation.get_nondominated_solutions import get_nondominated_solutions
from models.Hyperparameter import generate_hyperparameter
from models.Solution import Solution
from algorithms.EDA.FEDA.feda_executer import FEDAExecuter
from collections import defaultdict
=======
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from evaluation.get_nondominated_solutions import get_nondominated_solutions
from models.Solution import Solution
from algorithms.EDA.UMDA.umda_executer import UMDAExecuter
<<<<<<< HEAD
>>>>>>> 1e9aefc3 (Simple inititation of FEDA object, and comment added to explain the algorithm.)
=======
from collections import defaultdict
>>>>>>> 0a234eb2 (population is initiated, creating each individual following a topological order.)

import time
import numpy as np

<<<<<<< HEAD
<<<<<<< HEAD

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

    do

        2. Learning
        -- If X does not have any parents in graph structure, then learn its marginal probability
        -- If X does have parents in graph structure, learn Conditional: P(X| each Y in parents(X)==0) In the example above, P(2| 0==0,1==0).
        Thus, we only need to learn P(X|parents(X)) from requirements whose parents Y are not selected (if any of them were, then P(X) is fixed to 1).
        That is, we do not need P(X | any parents(X)==1), just the all parents(X)==0 case.
        This means that conditional probability can be stored in a unidimensional array,
         using the same array to store either marginal or conditional probability for each X.

        3. Sampling
        -- In the case of requirements without parents in graph structure, use learned marginal probability
        -- In any Y in parents(X) is set to 1, then X=1, else use P(X|parents(X)==0)

    while(!stop_criterion)
    """

    def __init__(self, execs, dataset_name: str = "p2", dataset: Dataset = None, random_seed: int = None, debug_mode: bool = False,
                 tackle_dependencies: bool = False,
                 population_length: int = 100, selection_scheme: str = "nds", max_generations: int = 100,
                 max_evaluations: int = 0, subset_size: int = 5):

        super().__init__(execs,dataset_name, dataset, random_seed, debug_mode, tackle_dependencies,
                         population_length, max_generations, max_evaluations, subset_size=subset_size)

        self.population = None
        self.selection_scheme: str = selection_scheme

        self.config_dictionary.update({'algorithm': 'feda'})

        self.hyperparameters.append(generate_hyperparameter(
            "selection_scheme", selection_scheme))
        self.config_dictionary['selection_scheme'] = selection_scheme

        self.executer = FEDAExecuter(algorithm=self, execs=execs)

        self.probs = np.full(self.dataset.num_pbis, 0)

        # self.graph[p]  returns list of children(p), if any.
        self.graph = defaultdict(list)
        self.parents_of = defaultdict(list)
        self.topological_order = self.compute_topological_order()
        self.orphans = self.find_orphans()

    def get_file(self) -> str:
        return (f"{str(self.__class__.__name__)}-{str(self.dataset_name)}-"
                f"{self.dependencies_to_string()}-{str(self.random_seed)}-{str(self.population_length)}-"
                f"{str(self.max_generations)}-{str(self.max_evaluations)}-"
                f"{str(self.selection_scheme)}.txt")
=======
""" Fixed structure EDA
Given an initial set of requirements (pbis) dependencies in the form of 
X1-->X2, FEDA uses this knowledge as a prefixed structure. 
e.g: we can have an acyclic graph like this: G={1-->3, 2-->3, 4, 3-->5}, where pbis 1,2,4 do not have parents, parents(3)={1,2} and parents(5)={3}.
 
Thus, learning is not structural and only applies to data.
Sampling is always performed following an ancestral order ([4,1,2,3,5] in the example above).

Algorithm is as follows:

1. Sampling of First generation:
-- In the case of requirements without parents,  then  P(X)=1/self.dataset.num_pbis
-- If any Y in parents(X) is set to 1, then X=1, else use P(X)=1/self.dataset.num_pbis

2. Learning
-- Marginal probability in the case of requirements without parents.
-- Conditional: P(X| each Y in parents(X)==0) 
In the example above, P(3| 1==0,2==0). thus, we only need to learn the prob. of X when from individuals where all parents of X are set to 0. 

3. Sampling
-- In the case of requirements without parents, use marginal probability 
-- In any Y in parents(X) is set to 1, then X=1, else use P(X|parents(X)==0)
"""

=======
>>>>>>> 0a234eb2 (population is initiated, creating each individual following a topological order.)

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
>>>>>>> 1e9aefc3 (Simple inititation of FEDA object, and comment added to explain the algorithm.)

        # self.probability_model = np.full(self.dataset.num_pbis, 1/self.dataset.num_pbis)

        self.graph = defaultdict(list)  # self.graph[p]  returns list of children(p), if any.
        self.parents_of = defaultdict(list)
        self.topological_order = self.compute_topological_order()
        self.orphans = self.find_orphans()


    def get_name(self) -> str:
        return (f"FEDA{str(self.population_length)}+{str(self.max_generations)}+"
                f"{str(self.max_evaluations)}")

<<<<<<< HEAD
<<<<<<< HEAD
    def df_find_data(self, df: any):
        return df[(df["Population Length"] == self.population_length) & (df["MaxGenerations"] == self.max_generations)
                  & (df["Selection Scheme"] == self.selection_scheme) & (df["Algorithm"] == self.__class__.__name__)
                  & (df["Dataset"] == self.dataset_name) & (df["MaxEvaluations"] == self.max_evaluations)
                  ]

    def run(self) -> Dict[str, Any]:
        self.reset()
        start = time.time()

        self.population = self.init_population()
        get_nondominated_solutions(self.population, self.nds)

        #plot_solutions(self.population)


        try:
            while not self.stop_criterion(self.num_generations, self.num_evaluations):
                # select individuals from self.population based on self.selection_scheme
                local_nds = self.select_individuals(self.population)
                #plot_solutions(local_nds)
                # learning

                self.probs = self.learn_probability_model(local_nds)


                # sampling
                #go = time.time()
                self.population = self.sample_new_population(self.probs)
                #print("Sampling new pop: ", time.time() - go)
               # plot_solutions(self.population)

                # evaluation  # update nds with solutions constructed and evolved in this iteration
                get_nondominated_solutions(self.population, self.nds) #TODO aquí se filtran las nds, y en la siguiente iteración también se filtran para local_nds! se hace doble?
                #plot_solutions(self.nds)
                self.num_generations += 1

                if self.debug_mode:
                    self.debug_data()
=======
    '''
        get string representation as X-->[list or requirements] for each requirement X
    '''
    def get_dependencies(self) -> str:
        deps = "Requirement Dependencies:\n"
        print(self.dataset.dependencies)
        for x in np.arange(len(self.dataset.dependencies)):
            deps = deps + f"{x}-->{self.dataset.dependencies[x]}\n"

        return deps

=======
>>>>>>> 0a234eb2 (population is initiated, creating each individual following a topological order.)
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
>>>>>>> 1e9aefc3 (Simple inititation of FEDA object, and comment added to explain the algorithm.)

        except EvaluationLimit:
            pass

        end = time.time()
<<<<<<< HEAD
        #plot_solutions(self.nds)
=======
>>>>>>> 1e9aefc3 (Simple inititation of FEDA object, and comment added to explain the algorithm.)

        print("\nNDS created has", self.nds.__len__(), "solution(s)")

        return {"population": self.nds,
                "time": end - start,
                "numGenerations": self.num_generations,
                "best_individual": self.best_individual,
                "numEvaluations": self.num_evaluations,
<<<<<<< HEAD
                "nds_debug": self.nds_debug,
                "population_debug": self.population_debug
                }

    '''
    1. Sampling of First generation:
        -- If X does not have any parents, then sample using  P(X)=1/self.dataset.num_pbis
        -- If any Y in parents(X) is set to 1, then X=1, else use P(X)=1/self.dataset.num_pbis
    '''

    def init_population(self) -> List[Solution]:

        population = []
        probs = np.full(self.dataset.pbis_score.size, 1 / self.dataset.pbis_score.size)

        for _ in np.arange(self.population_length):
            sample_selected = np.full(self.dataset.num_pbis, 0)
            # sample whole individual using P(X)= 1/self.dataset.num_pbis for all X
            replace = False # if True, less individuals do not reach cost=1
            while 1 not in sample_selected:
                sample_selected = np.random.choice(np.arange(self.dataset.num_pbis),
                                                   size=np.random.randint(self.dataset.num_pbis),
                                           replace=replace, p=probs) # np.random.binomial(1, probs)

            if replace: sample_selected = np.unique(sample_selected)
            # now follow topological order to check if any X must be set to 1
            for x in self.topological_order:
                for p in self.parents_of[x]:
                    if p in sample_selected and not x in sample_selected:
                        sample_selected = np.append(sample_selected, x)
            solution = Solution(self.dataset, None, selected=sample_selected)
            population.append(solution)


        #plot_solutions(population)
        return population

    '''
    2. Learning
    --case 1: If X does not have any parents in graph structure, then learn its marginal probability
    --case 2: P(X| each Y in parents(X)==0) In the example above, P(2| 0==0,1==0).
    Thus, we only need to learn P(X|parents(X)) from individuals where all Y in parents(X) are == 0.
    This means that conditional probability can be stored in an unidimensional array, using the same array that
    marginal probabilities, since P(X) is only computed once (marginal, or conditional).
    '''

    def learn_probability_model(self, individuals: List[Solution]) -> List[float]:

        # get values in numpy array to speed up learning
        vectors = []
        for sol in individuals:
            vectors.append(sol.selected)
        np_vectors = np.array(vectors)

        # case 1: marginal probs of X
        probs = np.sum(np_vectors, axis=0) / len(individuals)

        import warnings
        warnings.filterwarnings('error')
        # case 2: overwrite P(X|parents(X)==0) in the same probs array
        for x in np.arange(self.dataset.num_pbis):
            parents_x = self.parents_of[x]
            if len(parents_x) != 0:  # for each X with parents,
                subset = np_vectors
                # cumulative reduction of individuals filtering by having each Y in parents(X)==0
                for y in parents_x:
                    subset = subset[subset[:, y] == 0]
                # if len==0 (no individuals where all parents(X)=0) this lets marginal P(X) remain
                if len(subset) != 0:
                    # if len>0 (individuals with all parents(X)=0)
                    probs[x] = np.sum(subset[:, x]) / len(subset)
                    # overwrite P(X)
                # else: probs[x] = 0 TODO That is, do not sample X if there is no individual where all parents(X)==0

        return probs

    '''
    3. Sampling
    -- In the case of requirements without parents, use learned marginal probability
    -- In any Y in parents(X) is set to 1, then X=1, else use P(X|parents(X)==0)
    '''

    def sample_new_population(self, probs) -> List[Solution]:
        # init whole np 2d array with empty individuals
        population = np.zeros(
            shape=(self.population_length, self.dataset.num_pbis))

        # sample following topological order
        for x in self.topological_order:
            if x in self.orphans:  # create values for each orphan in all individuals at once
                x_values_in_pop = np.random.binomial(
                    n=1, p=probs[x], size=self.population_length)
                population[:, x] = x_values_in_pop
            else:  # create values for each x, in all individuals one by one
                parents_x = self.parents_of[x]
                for n in np.arange(self.population_length):  # for each individual
                    parent_is_set = False
                    for y in parents_x:  # find if any parent(X) is set to 1
                        parent_is_set = parent_is_set or population[n, y] == 1
                    if parent_is_set:
                        population[n, x] = 1  # then set X to 1
                    else:
                        population[n, x] = np.random.binomial(
                            n=1, p=probs[x], size=1)  # else use P(X|parents(X)==0)

        #  convert population into List of Solution
        new_population = []
        for individual in population:
            selected = np.where(individual == 1)
            new_population.append(
                Solution(self.dataset, None, selected=selected))

        return new_population

    def sample_new_population2(self, probs) -> List[Solution]:
        # init whole np 2d array with empty individuals
        population = np.zeros(
            shape=(self.population_length, self.dataset.num_pbis))

        # sample following topological order
        for x in self.topological_order:
            # first set x in all individual with its Prob computed (marginal or P(X|parents(X)==0)
            x_values_in_pop = np.random.binomial(
                    n=1, p=probs[x], size=self.population_length)
            population[:, x] = x_values_in_pop
            # now solve when any parent is set, then x must be 1
            parents_x = self.parents_of[x]
            if len(parents_x) > 0:
                p_values_in_pop = population[:, parents_x]
                for index, values in enumerate(p_values_in_pop):
                    if 1 in values:
                        population[index, x] = 1




        #  convert population into List of Solution
        new_population = []
        for individual in population:
            selected = np.where(individual == 1)
            new_population.append(
                Solution(self.dataset, None, selected=selected))

        return new_population

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

        # print("Dependencies Graph is: ", self.graph)
        # Mark all the vertices as not visited
        visited = [False] * v
        order = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(v):
            if not visited[i]:
                self.topological_sort_util(i, visited, order)

        # print("A topological order of graph structure is: ", order)

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
    feda = FEDAAlgorithm(dataset_name="2")
    feda.run()
=======
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
<<<<<<< HEAD

    print(feda.get_dependencies())
>>>>>>> 1e9aefc3 (Simple inititation of FEDA object, and comment added to explain the algorithm.)
=======
    feda.run()
>>>>>>> 0a234eb2 (population is initiated, creating each individual following a topological order.)
