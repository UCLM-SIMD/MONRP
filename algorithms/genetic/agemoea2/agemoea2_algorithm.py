import random
from typing import Any, Dict

import pymoo.algorithms.hyperparameters
from pymoo.core.repair import NoRepair
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling

import evaluation
from algorithms.genetic.abstract_genetic.abstract_genetic_algorithm import AbstractGeneticAlgorithm
from algorithms.genetic.agemoea2.agemoea_executer import AGEMOEAExecuter
from pymoo.algorithms.moo.age import AGEMOEA
import time

from models.problem_monrp import MONRProblem
from models.repair_pymoo import RepairPymoo
from datasets import Dataset
from evaluation.get_nondominated_solutions import get_nondominated_solutions

from models.Solution import Solution

pymoo.algorithms.hyperparameters.MultiRun


class AGEMOEA2Algorithm(AbstractGeneticAlgorithm):
    """https://pymoo.org/algorithms/moo/age2.html
    """

    def __init__(self, execs, dataset_name="test", dataset: Dataset = None, random_seed=None, population_length=20,
                 max_generations=1000, debug_mode=False, tackle_dependencies=True, subset_size=5,
                 sss_type=0, repair_deps = False, sss_per_iteration=False):

        super().__init__(execs, dataset_name, dataset, random_seed=random_seed, debug_mode=debug_mode, tackle_dependencies=tackle_dependencies,
                         population_length=population_length, max_generations=max_generations, max_evaluations=0,
                         subset_size=subset_size,  sss_type=sss_type, sss_per_iteration=sss_per_iteration)

        #if False, pymoo uses feasibility first, and at the end we discard unfeasible individuals. if true,
        # then individuals are repaired per iteration
        self.repair_deps = repair_deps
        self.executer = AGEMOEAExecuter(algorithm=self, execs=execs)

        self.population = None

        random.seed(self.random_seed)

        self.num_evaluations: int = 0
        self.num_generations: int = 0
        self.best_individual = None


        self.config_dictionary.update({'algorithm': 'agemoea2'})
        self.config_dictionary.update({'dependencies': 'True'})

        self.config_dictionary['population_length'] = population_length
        self.config_dictionary['max_generations'] = max_generations
        self.config_dictionary['max_evaluations'] = 0
        self.config_dictionary['repair_deps'] = self.repair_deps

    def get_file(self) -> str:
        return None

    def get_name(self) -> str:
       return None

    def reset(self) -> None:
        super().reset()

    def add_evaluation(self, new_population):
        pass

    def run(self) -> Dict[str, Any]:
        self.reset()
        nds_update_time = 0
        sss_total_time = 0
        start = time.time()

        #### ASK and TELL PROBLEM-DEPENDENT EXECUTION pymoo mode https://pymoo.org/algorithms/usage.html  ###

        # create problem representation
        count_deps = 0
        if self.tackle_dependencies:
            count_deps = sum(len(dep) for dep in self.dataset.dependencies if dep is not None)

        problem = MONRProblem(num_pbis=self.dataset.num_pbis, costs=self.dataset.pbis_cost_scaled,
                              satisfactions=self.dataset.pbis_satisfaction_scaled,
                              dependencies=self.dataset.dependencies, num_deps=count_deps)

        # create the pymoo algorithm object
        #by default, AGEMOE would use Feasbility First, that is, fill pop is unfeasible individuals if
        # we do not want that. we will repair them (repair parameter).
        repair_type = NoRepair() if not self.repair_deps else RepairPymoo()
        algorithm = AGEMOEA(pop_size=self.population_length, seed=random.randint(0, 99999), repair=repair_type,
                            sampling=BinaryRandomSampling(), mutation=BitflipMutation(prob=0.5))

        algorithm.setup(problem=problem, termination=('n_gen', self.max_generations))

        while algorithm.has_next():
            # ask the algorithm to create the next population (mutates, tournament, ...)
            pop = algorithm.ask()

            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            algorithm.evaluator.eval(problem, pop)

            # update nds
            update_start = time.time()
            current_pop=[]
            for ind in pop:
                sol = Solution(dataset=self.dataset, selected=ind.X, probabilities=None)
                current_pop.append(sol)
            get_nondominated_solutions(current_pop, self.nds)
            nds_update_time = nds_update_time + (time.time() - update_start)

            if self.sss_per_iteration:
                sss_start = time.time()
                self.nds = evaluation.solution_subset_selection.search_solution_subset(self.sss_type,
                                                                                       self.subset_size, self.nds)
                sss_total_time = sss_total_time + (time.time() - sss_start)

            # set the evaluated population as the one to use to create offspring
            algorithm.tell(infills=pop)

            # do some more things, printing, logging, storing or even modifying the algorithm object
            print(algorithm.n_gen, algorithm.evaluator.n_eval)

        #if dependencies are not repaired per iteration, they are repaired at the end of execution
        if not self.repair_deps and self.tackle_dependencies:
            self.nds = self.repair_population_dependencies(self.nds)

        end = time.time()
        print(end-start, "secs")
        

        return {"population": self.nds,
                "time": end - start,
                "nds_update_time": nds_update_time,
                "sss_total_time": sss_total_time,
                "numGenerations": self.max_generations,
                "bestGeneration": None,
                "best_individual": None,
                "numEvaluations": algorithm.evaluator.n_eval,
                "nds_debug": self.nds_debug,
                "population_debug": self.population_debug
                }
