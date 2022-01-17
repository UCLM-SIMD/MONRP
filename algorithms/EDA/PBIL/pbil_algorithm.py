from typing import Any, Dict, List, Tuple
from algorithms.EDA.eda_algorithm import EDAAlgorithm
from evaluation.get_nondominated_solutions import get_nondominated_solutions
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from algorithms.EDA.PBIL.pbil_executer import PBILExecuter
from models.Solution import Solution

import time
import numpy as np


class PBILAlgorithm(EDAAlgorithm):
    """Population Based Incremental Learning
    """

    def __init__(self, dataset_name: str = "test", random_seed: int = None, debug_mode: bool = False, tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0,
                 learning_rate: float = 0.1, mutation_prob: float = 0.1, mutation_shift: float = 0.1):

        super().__init__(dataset_name, random_seed, debug_mode, tackle_dependencies,
                         population_length, max_generations, max_evaluations)

        self.executer = PBILExecuter(algorithm=self)

        self.learning_rate: float = learning_rate
        self.mutation_prob: float = mutation_prob
        self.mutation_shift: float = mutation_shift

        self.file: str = (f"{str(self.__class__.__name__)}-{str(dataset_name)}-{str(random_seed)}-{str(population_length)}-"
                          f"{str(max_generations)}-{str(max_evaluations)}-{str(learning_rate)}-{str(mutation_prob)}-{str(mutation_shift)}.txt")

    def get_name(self) -> str:
        return (f"PBIL+{self.population_length}+{self.max_generations}+{self.max_evaluations}+"
                f"{self.learning_rate}+{self.mutation_prob}+{self.mutation_shift}")

    def df_find_data(self, df: any):
        return df[(df["Population Length"] == self.population_length) & (df["MaxGenerations"] == self.max_generations)
                  & (df["Learning Rate"] == self.learning_rate) & (df["Mutation Probability"] == self.mutation_prob)
                  & (df["Algorithm"] == self.__class__.__name__) & (df["Mutation Shift"] == self.mutation_shift)
                  & (df["Dataset"] == self.dataset_name)
                  ]

    def initialize_probability_vector(self) -> np.ndarray:
        probabilities = np.full(self.dataset.pbis_score.size, 0.5)
        #probabilities = np.full(self.dataset.pbis_score.size, 1/self.dataset.pbis_score.size)

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
        #nds_pop = population
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
        self.reset()

        self.probability_vector = self.initialize_probability_vector()

        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                self.population = []

                self.population = self.sample_new_population(
                    self.probability_vector)

                # repair population if dependencies tackled:
                if(self.tackle_dependencies):
                    self.population = self.repair_population_dependencies(
                        self.population)

                self.evaluate(self.population, self.best_individual)

                max_sample = self.select_individuals(self.population)

                self.probability_vector = self.learn_probability_model(
                    self.probability_vector, max_sample)

                # update nds with solutions constructed and evolved in this iteration
                get_nondominated_solutions(self.population, self.nds)

                self.num_generations += 1

                if self.debug_mode:
                    self.debug_data()

        except EvaluationLimit:
            pass

        end = time.time()

        print("\nNDS created has", self.nds.__len__(), "solution(s)")

        return {"population": self.nds,
                "time": end - start,
                "numGenerations": self.num_generations,
                "best_individual": self.best_individual,
                "numEvaluations": self.num_evaluations,
                "nds_debug": self.nds_debug,
                "population_debug": self.population_debug
                }
