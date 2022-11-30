from typing import Any, Dict, List

import evaluation.solution_subset_selection
from algorithms.EDA.eda_algorithm import EDAAlgorithm
from algorithms.abstract_algorithm.abstract_algorithm import plot_solutions
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from datasets import Dataset
from evaluation.get_nondominated_solutions import get_nondominated_solutions
from models.Solution import Solution
from algorithms.EDA.UMDA.umda_executer import UMDAExecuter

import time
import numpy as np

from models.Hyperparameter import generate_hyperparameter


class UMDAAlgorithm(EDAAlgorithm):
    """Univariate Marginal Distribution Algorithm
    """

    def __init__(self, execs,dataset_name: str = "test", dataset: Dataset = None, random_seed: int = None, debug_mode: bool = False, tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0,
                 selected_individuals: int = 60, selection_scheme: str = "nds",
                 replacement_scheme: str = "replacement", subset_size: int = 20, sss_type=0, sss_per_it=False):

        super().__init__(execs,dataset_name, dataset, random_seed, debug_mode, tackle_dependencies,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
                         population_length, max_generations, max_evaluations, subset_size=subset_size)
=======
                         population_length, max_generations, max_evaluations)
>>>>>>> 19c7836f (ahora todos los resultados se almacenan en results.json con un id unico para cada conjunto de parametros de lanzamiento)
=======
                         population_length, max_generations, max_evaluations, subset_size=subset_size)
>>>>>>> 5efa3a53 (new hyperparameter created: subset_size used to choose a subset of solutions from the final set of solutions returned by the executed algorithm. Also, nsgaii is added in extract_postMetrics.py.)
=======
                         population_length, max_generations, max_evaluations, subset_size=subset_size,
                         sss_type=sss_type, sss_per_iteration=sss_per_it)
>>>>>>> d19d5435 (hyperparms. 'sss_per_iteration' and 'sss_type' added to control the solution subset selection process.)

        self.executer = UMDAExecuter(algorithm=self, execs=execs)

        self.selected_individuals: int = selected_individuals

        self.selection_scheme: str = selection_scheme
        self.replacement_scheme: str = replacement_scheme

        self.config_dictionary.update({'algorithm': 'umda'})

        self.hyperparameters.append(generate_hyperparameter(
            "selection_scheme", selection_scheme))
        self.config_dictionary['selection_scheme'] = selection_scheme
        self.hyperparameters.append(generate_hyperparameter(
            "replacement_scheme", replacement_scheme))
        self.config_dictionary['replacement_scheme'] = replacement_scheme

    def get_file(self) -> str:
        return (f"{str(self.__class__.__name__)}-{str(self.dataset_name)}-"
                f"{self.dependencies_to_string()}-{str(self.random_seed)}-{str(self.population_length)}-"
                f"{str(self.max_generations)}-{str(self.max_evaluations)}-{str(self.selected_individuals)}-{str(self.selection_scheme)}-"
                f"{str(self.replacement_scheme)}.txt")

    def get_name(self) -> str:
        return (f"UMDA{str(self.population_length)}+{str(self.max_generations)}+{str(self.max_evaluations)}+"
                f"{str(self.selected_individuals)}+{str(self.selection_scheme)}+{str(self.replacement_scheme)}")

    def df_find_data(self, df: any):
        return df[(df["Population Length"] == self.population_length) & (df["MaxGenerations"] == self.max_generations)
                  & (df["Selection Scheme"] == self.selection_scheme) & (df["Selected Individuals"] == self.selected_individuals)
                  & (df["Algorithm"] == self.__class__.__name__) & (df["Replacement Scheme"] == self.replacement_scheme)
                  & (df["Dataset"] == self.dataset_name) & (df["MaxEvaluations"] == self.max_evaluations)
                  ]

    def learn_probability_model(self, population: List[Solution]) -> List[float]:
        """Learns probability from a set of solutions, returning an array of probabilities for each gene to be 1.
        """
        probability_model = []
        # for each gene:
        for index in np.arange(len(self.dataset.pbis_cost_scaled)):
            num_ones = 0
            # count selected
            for individual in population:
                num_ones += individual.selected[index]
            # prob = nº 1s / nº total
            index_probability = num_ones/len(population)
            probability_model.append(index_probability)

        return probability_model

<<<<<<< HEAD
=======
    def generate_sample_from_probabilities_binomial(self, probabilities: List[float]) -> Solution:
        """Generates a sample given the probability vector, using numpy binomial method.
        """
        sample_selected = np.random.binomial(1, probabilities)
        sample = Solution(self.dataset, None, selected=sample_selected)
        return sample



    def generate_sample_from_probabilities(self, probabilities: List[float]) -> Solution:
        """Generates a sample given the probability vector, using scaled probabilities
        """
        probs = [prob * 10 for prob in probabilities]
        sum_probs = np.sum(probs)
        scaled_probs = probs / sum_probs
        sample = Solution(self.dataset, scaled_probs)
        return sample

    def replace_population_from_probabilities_elitism(self, probability_model: List[float], population: List[Solution]) -> List[Solution]:
        new_population = []
        # elitist R-1 inds
        for _ in np.arange(self.population_length-1):
            new_individual = self.generate_sample_from_probabilities_binomial(
                probability_model)
            # new_individual = self.generate_sample_from_probabilities(
            #    probability_model)
            new_population.append(new_individual)

        # elitism -> add best individual from old population
        population.sort(
            key=lambda x: x.compute_mono_objective_score(), reverse=True)
        new_population.append(population[0])

        return new_population

    def replace_population_from_probabilities(self, probability_model: List[float]) -> List[Solution]:
        new_population = []
        for _ in np.arange(self.population_length):
            new_individual = self.generate_sample_from_probabilities_binomial(
                probability_model)
            # new_individual = self.generate_sample_from_probabilities(
            #    probability_model)
            new_population.append(new_individual)

        return new_population

>>>>>>> 0a234eb2 (population is initiated, creating each individual following a topological order.)
    def sample_new_population(self, probability_model: List[float]) -> List[Solution]:
        """Given a probability vector, samples a new population depending on the scheme selected.
        """
        if self.replacement_scheme == "replacement":
            population = self.replace_population_from_probabilities(
                probability_model)
        elif self.replacement_scheme == "elitism":
            population = self.replace_population_from_probabilities_elitism(
                probability_model, self.population)
        return population

    def run(self) -> Dict[str, Any]:
        self.reset()
        start = time.time()
        nds_update_time = 0

        self.population = self.generate_initial_population()
<<<<<<< HEAD
<<<<<<< HEAD
        #plot_solutions(self.population)
        if (self.tackle_dependencies):
            self.population = self.repair_population_dependencies(
                self.population)
<<<<<<< HEAD
<<<<<<< HEAD
=======

=======
        #plot_solutions(self.population)
>>>>>>> a636b2d9 (error in initialization in FEDA solved)
        self.evaluate(self.population, self.best_individual)
>>>>>>> 73926cb9 (now satisfaction and cost are scaled such that all together sum up 1)
=======
        #plot_solutions(self.population)
        self.evaluate(self.population, self.best_individual)

>>>>>>> ecb85730 (now pbil and geneticnds keep nds from initial population, then pareto is now wider)
=======
>>>>>>> a7235ed3 (solved comments from pull request, added minor local changes in some files)
        get_nondominated_solutions(self.population, self.nds)
        #plot_solutions(self.population)



<<<<<<< HEAD

=======
>>>>>>> a7235ed3 (solved comments from pull request, added minor local changes in some files)


        if self.debug_mode:
            self.debug_data()

        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                # selection
                # TODO individuals = self.select_individuals(self.population+old_pop)
                individuals = self.select_individuals(self.population)

                # learning
                probability_model = self.learn_probability_model(
                    individuals)
                # old_pop = self.population.copy()

                # replacement
                self.population = self.sample_new_population(probability_model)
<<<<<<< HEAD
<<<<<<< HEAD

=======
                #plot_solutions(self.population)
>>>>>>> ecb85730 (now pbil and geneticnds keep nds from initial population, then pareto is now wider)
=======

>>>>>>> a7235ed3 (solved comments from pull request, added minor local changes in some files)
                # repair population if dependencies tackled:
                if(self.tackle_dependencies):
                    self.population = self.repair_population_dependencies(
                        self.population)
<<<<<<< HEAD
<<<<<<< HEAD

<<<<<<< HEAD
                # evaluation  # update nds with solutions constructed and evolved in this iteration
=======
=======
                #plot_solutions(self.population)
>>>>>>> ecb85730 (now pbil and geneticnds keep nds from initial population, then pareto is now wider)
                # evaluation
                self.evaluate(self.population, self.best_individual)

                # update nds with solutions constructed and evolved in this iteration
                #plot_solutions(self.population)
>>>>>>> 73926cb9 (now satisfaction and cost are scaled such that all together sum up 1)
=======

                # evaluation  # update nds with solutions constructed and evolved in this iteration
<<<<<<< HEAD
>>>>>>> a7235ed3 (solved comments from pull request, added minor local changes in some files)
=======
                update_start = time.time()
>>>>>>> f9ef1beb (total time used to update nds_archive is now measured)
                get_nondominated_solutions(self.population, self.nds)
                nds_update_time = nds_update_time + (time.time() - update_start)
                #plot_solutions(self.nds)
                self.num_generations += 1

                if self.sss_per_iteration:
                    self.nds = evaluation.solution_subset_selection.search_solution_subset(self.sss_type,
                                                                                           self.subset_size, self.nds)


                if self.debug_mode:
                    self.debug_data()

        except EvaluationLimit:
            pass

        end = time.time()
        #plot_solutions(self.nds)

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
