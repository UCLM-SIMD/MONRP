from typing import Any, Dict, List
from algorithms.EDA.eda_algorithm import EDAAlgorithm
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from evaluation.get_nondominated_solutions import get_nondominated_solutions
from models.Solution import Solution
from algorithms.EDA.UMDA.umda_executer import UMDAExecuter

import time
import numpy as np

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


class FEDAAlgorithm(EDAAlgorithm):

    def __init__(self, dataset_name: str = "1", random_seed: int = None, debug_mode: bool = False,
                 tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0):

        super().__init__(dataset_name, random_seed, debug_mode, tackle_dependencies,
                         population_length, max_generations, max_evaluations)

        self.population = None
        self.executer = UMDAExecuter(algorithm=self)

        self.file: str = (
            f"{str(self.__class__.__name__)}-{str(dataset_name)}-{str(random_seed)}-{str(population_length)}-"
            f"{str(max_generations)}-{str(max_evaluations)}.txt")

    def get_name(self) -> str:
        return (f"FEDA{str(self.population_length)}+{str(self.max_generations)}+"
                f"{str(self.max_evaluations)}")

    '''
        get string representation as X-->[list or requirements] for each requirement X
    '''
    def get_dependencies(self) -> str:
        deps = "Requirement Dependencies:\n"
        print(self.dataset.dependencies)
        for x in np.arange(len(self.dataset.dependencies)):
            deps = deps + f"{x}-->{self.dataset.dependencies[x]}\n"

        return deps

    def run(self) -> Dict[str, Any]:
        self.reset()
        paretos = []
        start = time.time()

        # el modelo de probabilidad inicial es fijado por la estructura previa de dependencias conocidas
        #  probability_model = self.learn_probability_model(individuals)

        # self.population = self.sample_new_population(probability_model)
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


if __name__ == "__main__":
    feda = FEDAAlgorithm(dataset_name="test")

    print(feda.get_dependencies())
