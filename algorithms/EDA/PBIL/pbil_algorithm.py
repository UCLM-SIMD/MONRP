from typing import Any, Dict, List, Tuple
from algorithms.EDA.eda_algorithm import EDAAlgorithm
from algorithms.abstract_algorithm.abstract_algorithm import plot_solutions
from datasets import Dataset
from evaluation.get_nondominated_solutions import get_nondominated_solutions
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from algorithms.EDA.PBIL.pbil_executer import PBILExecuter
from models.Solution import Solution

import time
import numpy as np

from models.Hyperparameter import generate_hyperparameter


class PBILAlgorithm(EDAAlgorithm):
    """Population Based Incremental Learning
    """

    def __init__(self,execs, dataset_name: str = "test", dataset: Dataset = None, random_seed: int = None, debug_mode: bool = False, tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0,
<<<<<<< HEAD
<<<<<<< HEAD
                 learning_rate: float = 0.1, mutation_prob: float = 0.1,
                 mutation_shift: float = 0.1, subset_size: int = 5):

        super().__init__(execs,dataset_name, dataset, random_seed, debug_mode, tackle_dependencies,
                         population_length, max_generations, max_evaluations, subset_size=subset_size)
=======
                 learning_rate: float = 0.1, mutation_prob: float = 0.1, mutation_shift: float = 0.1, ):

        super().__init__(execs,dataset_name, dataset, random_seed, debug_mode, tackle_dependencies,
                         population_length, max_generations, max_evaluations)
>>>>>>> 19c7836f (ahora todos los resultados se almacenan en results.json con un id unico para cada conjunto de parametros de lanzamiento)
=======
                 learning_rate: float = 0.1, mutation_prob: float = 0.1,
                 mutation_shift: float = 0.1, subset_size: int = 5):

        super().__init__(execs,dataset_name, dataset, random_seed, debug_mode, tackle_dependencies,
                         population_length, max_generations, max_evaluations, subset_size=subset_size)
>>>>>>> 5efa3a53 (new hyperparameter created: subset_size used to choose a subset of solutions from the final set of solutions returned by the executed algorithm. Also, nsgaii is added in extract_postMetrics.py.)

        self.executer = PBILExecuter(algorithm=self, execs=execs)

        self.learning_rate: float = learning_rate
        self.mutation_prob: float = mutation_prob
        self.mutation_shift: float = mutation_shift
        self.config_dictionary.update({'algorithm': 'pbil'})

        self.hyperparameters.append(generate_hyperparameter(
            "learning_rate", learning_rate))
        self.config_dictionary['learning_rate'] = learning_rate
        self.hyperparameters.append(generate_hyperparameter(
            "mutation_prob", mutation_prob))
        self.config_dictionary['mutation_prob'] = mutation_prob
        self.hyperparameters.append(generate_hyperparameter(
            "mutation_shift", mutation_shift))
        self.config_dictionary['mutation_shift'] = mutation_shift

    def get_file(self) -> str:
        return (f"{str(self.__class__.__name__)}-{str(self.dataset_name)}-"
                f"{self.dependencies_to_string()}-{str(self.random_seed)}-{str(self.population_length)}-"
                f"{str(self.max_generations)}-{str(self.max_evaluations)}-{str(self.learning_rate)}-"
                f"{str(self.mutation_prob)}-{str(self.mutation_shift)}.txt")

    def get_name(self) -> str:
        return (f"PBIL+{self.population_length}+{self.max_generations}+{self.max_evaluations}+"
                f"{self.learning_rate}+{self.mutation_prob}+{self.mutation_shift}")

    def df_find_data(self, df: any):
        return df[(df["Population Length"] == self.population_length) & (df["MaxGenerations"] == self.max_generations)
                  & (df["Learning Rate"] == self.learning_rate) & (df["Mutation Probability"] == self.mutation_prob)
                  & (df["Algorithm"] == self.__class__.__name__) & (df["Mutation Shift"] == self.mutation_shift)
                  & (df["Dataset"] == self.dataset_name) & (df["MaxEvaluations"] == self.max_evaluations)
                  ]

    def initialize_probability_vector(self) -> np.ndarray:
        #probabilities = np.full(self.dataset.pbis_score.size, 0.5)
        probabilities = np.full(self.dataset.pbis_score.size, 1/self.dataset.pbis_score.size)

        return probabilities

    def select_individuals(self, population: List[Solution]) -> Solution:
        """Select best individual TODO choose the method used (mo or nds) depending on config
        """
        max_sample = self.find_max_sample_nds(
            population, self.nds)
        # max_sample = self.find_max_sample_pop(
        #    population)
        return max_sample

    def find_max_sample_monoscore(self, population: List[Solution]) -> Tuple[float, Solution]:
        population.sort(
            key=lambda x: x.compute_mono_objective_score(), reverse=True)
        max_score = population[0].compute_mono_objective_score()
        max_sample = population[0]
        return max_score, max_sample

    def find_max_sample_nds(self, population: List[Solution], nds) -> Solution:
        if len(nds) > 0:
            random_index = np.random.randint(len(nds))
            return nds[random_index]
        else:
            random_index = np.random.randint(len(population))
            return population[random_index]

    def find_max_sample_pop(self, population: List[Solution]) -> Solution:
        nds_pop = get_nondominated_solutions(population, [])
        # nds_pop = population
        random_index = np.random.randint(len(nds_pop))
        return nds_pop[random_index]

    def learn_probability_model(self, probability_vector: np.ndarray, max_sample: Solution) -> np.ndarray:
        """Updates the probability vector using the sample given
        """
        for i in np.arange(len(probability_vector)):
            probability_vector[i] = probability_vector[i]*(
                1-self.learning_rate)+max_sample.selected[i]*(self.learning_rate)

        for i in np.arange(len(probability_vector)):
            prob = np.random.random_sample()
            if prob < self.mutation_prob:
                probability_vector[i] = probability_vector[i]*(
                    1-self.mutation_shift) + (np.random.randint(2))*self.mutation_shift
        return probability_vector

    def sample_new_population(self, probability_model: List[float]) -> List[Solution]:
        """Given a probability vector, samples a new population depending on the scheme selected.
        """
        population = self.replace_population_from_probabilities(
            probability_model)
        return population

    def run(self) -> Dict[str, Any]:
        start = time.time()
        nds_update_time = 0
        self.reset()

        self.probability_vector = self.initialize_probability_vector()

        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                self.population = []

                self.population = self.sample_new_population(
                    self.probability_vector)
                #plot_solutions(self.population)

                # repair population if dependencies tackled:
                if(self.tackle_dependencies):
                    self.population = self.repair_population_dependencies(
                        self.population)
                #plot_solutions(self.population)
<<<<<<< HEAD
<<<<<<< HEAD

=======
                self.evaluate(self.population, self.best_individual)
>>>>>>> ecb85730 (now pbil and geneticnds keep nds from initial population, then pareto is now wider)
=======

>>>>>>> a7235ed3 (solved comments from pull request, added minor local changes in some files)

                max_sample = self.select_individuals(self.population)

                self.probability_vector = self.learn_probability_model(
                    self.probability_vector, max_sample)

                # update nds with solutions constructed and evolved in this iteration
                update_start = time.time()
                get_nondominated_solutions(self.population, self.nds)
                nds_update_time = nds_update_time + (time.time() - update_start)
                #plot_solutions(self.nds)
                self.num_generations += 1

                if self.debug_mode:
                    self.debug_data()

        except EvaluationLimit:
            pass
<<<<<<< HEAD
<<<<<<< HEAD
        #plot_solutions(self.population)
=======
        plot_solutions(self.population)
>>>>>>> ecb85730 (now pbil and geneticnds keep nds from initial population, then pareto is now wider)
=======
        #plot_solutions(self.population)
>>>>>>> d98580f6 (solved issue in mimic when sampling individuals)
        end = time.time()

        print("\nNDS created has", self.nds.__len__(), "solution(s)")

        return {"population": self.nds,
                "time": end - start,
                "nds_update_time": nds_update_time,
                "numGenerations": self.num_generations,
                "best_individual": self.best_individual,
                "numEvaluations": self.num_evaluations,
                "nds_debug": self.nds_debug,
                "population_debug": self.population_debug
                }
