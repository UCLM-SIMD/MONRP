import random
from typing import Any, Dict, List, Tuple
from algorithms.EDA.bivariate.MIMIC.mimic_executer import MIMICExecuter
from algorithms.EDA.eda_algorithm import EDAAlgorithm
from algorithms.abstract_algorithm.abstract_algorithm import plot_solutions
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from datasets import Dataset
from evaluation.get_nondominated_solutions import get_nondominated_solutions
from models.Solution import Solution
from models.Hyperparameter import generate_hyperparameter
import time
import numpy as np
import math
from scipy import stats as scipy_stats


class MIMICAlgorithm(EDAAlgorithm):

    def __init__(self, dataset_name: str = "test", dataset: Dataset = None, random_seed: int = None,
                 debug_mode: bool = False, tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0,
                 selected_individuals: int = 60, selection_scheme: str = "nds", replacement_scheme: str = "replacement",
                 execs=10, subset_size: int = 10):


        super().__init__(execs=execs, dataset_name=dataset_name, random_seed=random_seed,debug_mode=debug_mode,
                         tackle_dependencies=tackle_dependencies,population_length=population_length,
<<<<<<< HEAD
<<<<<<< HEAD
                         subset_size=subset_size, max_generations=max_generations,  max_evaluations= max_evaluations)
=======
                         subset_size=subset_size, max_generations=max_generations)
>>>>>>> 80ff396a (mimic added to execution framework. swapping of unfr and gdplus in jupyter notebook solved.)
=======
                         subset_size=subset_size, max_generations=max_generations,  max_evaluations= max_evaluations)
>>>>>>> d98580f6 (solved issue in mimic when sampling individuals)

        self.executer = MIMICExecuter(algorithm=self, execs=execs)

        self.gene_size: int = len(self.dataset.pbis_cost)
        print(self.hyperparameters)
        self.selected_individuals: int =\
            selected_individuals if selected_individuals<=population_length else population_length


        self.selection_scheme: str = selection_scheme
        self.replacement_scheme: str = replacement_scheme

        self.config_dictionary.update({'algorithm': 'mimic'})


        self.hyperparameters.append(generate_hyperparameter(
            "selected_individuals", selected_individuals))
        self.config_dictionary['selected_individuals'] = selected_individuals

        self.hyperparameters.append(generate_hyperparameter(
            "selection_scheme", selection_scheme))
        self.config_dictionary['selection_scheme'] = selection_scheme

        self.hyperparameters.append(generate_hyperparameter(
            "replacement_scheme", replacement_scheme))
        self.config_dictionary['replacement_scheme'] = replacement_scheme

        self.population: List[Solution] = []

    def get_file(self) -> str:
        return (f"{str(self.__class__.__name__)}-{str(self.dataset_name)}-"
                f"{self.dependencies_to_string()}-{str(self.random_seed)}-{str(self.population_length)}-"
                f"{str(self.max_generations)}-{str(self.max_evaluations)}-{str(self.selected_individuals)}-{str(self.selection_scheme)}-"
                f"{str(self.replacement_scheme)}.txt")

    def get_name(self) -> str:
        return (f"MIMIC+{self.population_length}+{self.max_generations}+"
                f"{self.max_evaluations}")

    def df_find_data(self, df: any):
        return df[(df["Population Length"] == self.population_length) & (df["MaxGenerations"] == self.max_generations)
                  & (df["Selection Scheme"] == self.selection_scheme) & (df["Selected Individuals"] == self.selected_individuals)
                  & (df["Algorithm"] == self.__class__.__name__) & (df["Replacement Scheme"] == self.replacement_scheme)
                  & (df["Dataset"] == self.dataset_name) & (df["MaxEvaluations"] == self.max_evaluations)
                  ]

    def learn_probability_model(self, population: List[Solution], selected_individuals: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # init structures
        parents = np.zeros(self.gene_size, dtype=int)
        used = np.full(self.gene_size, False)
        variables = np.zeros(self.gene_size, dtype=int)
        conditionals = np.zeros((self.gene_size, 2), dtype=float)

        marginals = self.learn_marginals(population, selected_individuals)

        # Obtain entropies.
        entropies = np.zeros(self.gene_size)
        for i in range(self.gene_size):
            entropies[i] = self.get_entropy(
                population, i, selected_individuals)

        # Takes the variable with less entropy as the first.
        current_var = np.argmin(entropies)
        parents[0] = -1
        variables[0] = current_var

        # Marks it as used.
        used[current_var] = True

        # Adds iteratively the variable with less conditional entropy.
        for i in range(1, self.gene_size):
            # Chooses the next variable.
            parents[i] = current_var
            current_var = self.get_lower_conditional_entropy(
                population, current_var, used, selected_individuals)
            variables[i] = current_var
            used[current_var] = True
            prob_x, prob_y, prob_xy = self.get_distributions(
                population, current_var, parents[i], selected_individuals)
            conditionals[i][0] = prob_xy[1][0]
            conditionals[i][1] = prob_xy[1][1]

        return marginals, parents, variables, conditionals

    def learn_marginals(self, population: List[Solution], selected_individuals: int, laplace: int = 0):
        marginals = np.zeros(self.gene_size)
        # if fixed number -> self.selected_individuals. if selection by NDS ->unknown ->len
        #selected_individuals = len(population)
        for i in range(selected_individuals):
            for j in range(self.gene_size):
                if population[i].selected[j] == 1:
                    marginals[j] += 1
        for j in range(self.gene_size):
            marginals[j] = (marginals[j]+laplace) / \
                (selected_individuals+(2*laplace))
        return marginals

    def get_probability_distribution(self, elements: List[Solution], v1: int, N: int, laplace: int = 1) -> np.ndarray:
        prob = np.zeros(2)
        for i in range(N):
            prob[elements[i].selected[v1]] += 1.0
        for i in range(2):
            if laplace == 1:
                prob[i] = (prob[i]+1)/N+2
            else:
                prob[i] = (prob[i])/N
        return prob

    def get_entropy(self, elements: List[Solution], var1: int, N: int) -> float:
        probs = self.get_probability_distribution(elements, var1, N, 0)
        return scipy_stats.entropy(probs, base=2)

    def get_conditional_entropy(self, population: List[int], var1: int, var2: int, N: int) -> float:
        entropy: float = 0.0
        prob_x, prob_y, prob_xy = self.get_distributions(
            population, var1, var2, N, 1)
        for j in range(2):
            entropy2 = 0.0
            for i in range(2):
                if(prob_xy[i][j] > 0):
                    entropy2 += prob_xy[i][j]*math.log2(prob_xy[i][j])
            if entropy2 != 0:
                entropy2 *= -1
            entropy += prob_y[j]*entropy2
        return entropy

    def get_lower_conditional_entropy(self, population: List[Solution], parent: int, used: List[bool], N: int) -> int:
        index: int = -1
        min_ce = float("inf")
        for i in range(self.gene_size):
            if(used[i]):
                continue
            ce = self.get_conditional_entropy(population, parent, i, N)
            if(ce < min_ce):
                min_ce = ce
                index = i
        return index

    def get_distributions(self, population: List[Solution], X: int, Y: int, N: int, laplace: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_y = np.zeros(2)
        prob_y = np.zeros(2)
        prob_x = np.zeros(2)
        prob_xy = np.zeros((2, 2))
        for row in range(N):
            prob_x[population[row].selected[X]] += 1
            prob_y[population[row].selected[Y]] += 1
            prob_xy[population[row].selected[X]
                    ][population[row].selected[Y]] += 1
            num_y[population[row].selected[Y]] += 1

        for i in range(2):
            if laplace == 1:
                prob_x[i] = (prob_x[i]+1.0)/(N+2)
            else:
                prob_x[i] = prob_x[i]/N
            for j in range(2):
                if laplace == 1:
                    prob_xy[i][j] = (prob_xy[i][j]+1.0)/(num_y[j]+2)
                else:
                    prob_xy[i][j] = prob_xy[i][j]/num_y[j]

        for i in range(2):
            if laplace == 1:
                prob_y[i] = (prob_y[i]+1.0)/(N+2)
            else:
                prob_y[i] = prob_y[i]/N

        return prob_x, prob_y, prob_xy

    def sample_new_population(self, marginals: List[float], parents: List[int], variables: List[int], conditionals: List[List[float]]) -> List[Solution]:
        new_population = []
        for _ in np.arange(self.population_length):
            new_individual = self.generate_sample(
                marginals, parents, variables, conditionals)
            new_population.append(new_individual)
        return new_population

    def generate_sample(self, marginals: List[float], parents: List[int], variables: List[int], conditionals: List[List[float]]) -> Solution:
        sample = np.zeros(self.gene_size, dtype=int)
        for j in range(self.gene_size):
            if(parents[j] == -1):
                if(random.random() < marginals[variables[j]]):
                    sample[variables[j]] = 1
                else:
                    sample[variables[j]] = 0

            else:
                if(random.random() < conditionals[j][sample[parents[j]]]):
                    sample[variables[j]] = 1
                else:
                    sample[variables[j]] = 0
        selected = np.where(sample == 1)
        sample_ind = Solution(self.dataset, None, selected=selected)
        return sample_ind

    def run(self) -> Dict[str, Any]:
        self.reset()
        start = time.time()

        self.population = self.generate_initial_population()

        get_nondominated_solutions(self.population, self.nds)



        if self.debug_mode:
            self.debug_data()

        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                # selection
                individuals = self.select_individuals(self.population)


                # learning
                marginals, parents, variables, conditionals = self.learn_probability_model(
                    individuals, len(individuals))

                # replacement
                self.population = self.sample_new_population(
                    marginals, parents, variables, conditionals)


                # repair population if dependencies tackled:
                if(self.tackle_dependencies):
                    self.population = self.repair_population_dependencies(
                        self.population)

                # evaluation # update nds with solutions constructed and evolved in this iteration
                get_nondominated_solutions(self.population, self.nds)

                self.num_generations += 1

                if self.debug_mode:
                    self.debug_data()

        except EvaluationLimit:
            pass

        end = time.time()
        #plot_solutions(self.nds)

        print("\nNDS created has", self.nds.__len__(), "solution(s)")

        return {"population": self.nds,
                "time": end - start,
                "numGenerations": self.num_generations,
                "best_individual": self.best_individual,
                "numEvaluations": self.num_evaluations,
                "nds_debug": self.nds_debug,
                "population_debug": self.population_debug
                }
