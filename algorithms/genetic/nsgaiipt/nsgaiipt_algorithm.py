import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from algorithms.genetic.nsgaiipt.nsgaiipt_executer import NSGAIIPTExecuter

import evaluation
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from algorithms.genetic.abstract_genetic.abstract_genetic_algorithm import AbstractGeneticAlgorithm
from algorithms.genetic.nsgaii.nsgaii_executer import NSGAIIExecuter
import copy
import time
from datasets import Dataset

from models.Solution import Solution


class NSGAIIPTAlgorithm(AbstractGeneticAlgorithm):
    """Nondominated Sorting Genetic Algorithm II PT, as defined in paper "An Effective Method of Systems Requirement Optimization Based on Genetic Algorithms": 10.12785/isl/060102.
    """

    def __init__(self, execs, dataset_name="test", dataset: Dataset = None, random_seed=None, population_length=20,
                 max_generations=1000, max_evaluations=0,
                 selection="tournament", selection_candidates=2,
                 crossover="onepoint", crossover_prob=0.9,
                 mutation="flipeachbit", mutation_prob=0.1,
                 debug_mode=False, tackle_dependencies=False, subset_size=5, replacement='elitism',
                 sss_type=0, sss_per_it=False):

        super().__init__(execs, dataset_name, dataset, random_seed, debug_mode, tackle_dependencies,
                         population_length, max_generations, max_evaluations, subset_size=subset_size,
                         sss_type=sss_type, sss_per_iteration=sss_per_it)

        self.executer = NSGAIIPTExecuter(algorithm=self, execs=execs)
        self.selection_scheme = selection
        self.selection_candidates = selection_candidates
        self.crossover_scheme = crossover
        self.crossover_prob = crossover_prob
        self.mutation_scheme = mutation
        self.mutation_prob = mutation_prob
        self.replacement_scheme = "elitism"

        self.population = None
        self.best_generation_avgValue = None
        self.best_generation = None

        self.num_evaluations: int = 0
        self.num_generations: int = 0
        self.best_individual = None

        self.config_dictionary.update({'algorithm': 'nsgaiipt'})
        self.config_dictionary['population_length'] = population_length
        self.config_dictionary['max_generations'] = max_generations
        self.config_dictionary['max_evaluations'] = max_evaluations
        self.config_dictionary['selection_candidates'] = selection_candidates
        self.config_dictionary['crossover_prob'] = crossover_prob
        self.config_dictionary['mutation_prob'] = mutation_prob

        if selection == "tournament":
            self.selection = self.selection_tournament
        self.config_dictionary['selection'] = selection

        if crossover == "onepoint":
            self.crossover = self.crossover_one_point
        self.config_dictionary['crossover'] = crossover

        if mutation == "flip1bit":
            self.mutation = self.mutation_flip1bit
        elif mutation == "flipeachbit":
            self.mutation = self.mutation_flipeachbit
        self.config_dictionary['mutation'] = mutation

        self.config_dictionary['replacement'] = self.replacement_scheme

    def get_file(self) -> str:
        return (f"{str(self.__class__.__name__)}-{str(self.dataset_name)}-"
                f"{self.dependencies_to_string()}-{str(self.random_seed)}-{str(self.population_length)}-"
                f"{str(self.max_generations)}-{str(self.max_evaluations)}-"
                f"{str(self.selection_scheme)}-{str(self.selection_candidates)}-"
                f"{str(self.crossover_scheme)}-{str(self.crossover_prob)}-{str(self.mutation_scheme)}-"
                f"{str(self.mutation_prob)}-{str(self.replacement_scheme)}.txt")

    def get_name(self) -> str:
        return f"NSGA-IIPT{str(self.population_length)}+{str(self.max_generations)}+{str(self.max_evaluations)}+{str(self.crossover_prob)}\
            +{str(self.mutation_scheme)}+{str(self.mutation_prob)}"

    def add_evaluation(self, new_population: List[Solution]) -> None:
        """Handles evaluation count, finishing algorithm execution if stop criterion is met by raising an exception.
        """
        self.num_evaluations += 1
        if self.stop_criterion(self.num_generations, self.num_evaluations):
            new_population, fronts = self.fast_nondominated_sort(
                new_population)
            self.best_generation, self.best_generation_avgValue = self.calculate_last_generation_with_enhance(
                self.best_generation, self.best_generation_avgValue, self.num_generations,
                new_population)
            raise EvaluationLimit

    def reset(self) -> None:
        super().reset()

    def run(self) -> Dict[str, Any]:
        self.reset()
        start = time.time()
        nds_update_time = 0

        # init nsgaii
        self.population = self.generate_starting_population()
        self.evaluate(self.population, self.best_individual)

        # 4 Correct initial population(p)
        if (self.tackle_dependencies):
            self.population = self.repair_population_dependencies(
                self.population)
            #fronts[0] = self.repair_population_dependencies(
                #fronts[0])

        # 5 Rank=non dominated sorting(P)
        # 6 CD=crowding distance assignment(P)
        # 7 Sort(P,Rank,CD)
        self.population, fronts = self.fast_nondominated_sort(self.population)
        self.calculate_crowding_distance(self.population)

        try:
            # 8 While not Termination Condition() Do
            while not self.stop_criterion(self.num_generations, self.num_evaluations):
                # 9 parents ← selection(p);
                offsprings = self.selection(self.population)

                # 10 Pc← Crossover(pc,parents);
                offsprings_c = self.crossover(offsprings)

                # 11 Pm← Mutation(pm,offspring);
                offsprings_m = self.mutation(offsprings_c)

                # 12 P← Merge(P,Pc,Pm);
                self.population = offsprings_c
                self.population.extend(offsprings_m)

                # 14 Remove duplicate individul(P) # TODO should go before separating in fronts
                self.population = self.remove_duplicates(self.population)

                #repair poulation
                if (self.tackle_dependencies):
                    self.population = self.repair_population_dependencies(
                        self.population)

                # 13 Sort(P,Rank,CD)
                self.evaluate(self.population, self.best_individual)
                self.population, fronts = self.fast_nondominated_sort(
                    self.population)
                for front in fronts:
                    self.calculate_crowding_distance(front)

                # 15 If size(P<pop size)
                if len(self.population) < self.population_length:
                    # 16 Generate(R,pop size-size(P))
                    # TODO check crowding distance of elems in self.pop is OK
                    offsprings = self.selection(self.population)
                    offsprings = self.crossover(offsprings)
                    offsprings = self.mutation(offsprings)

                    # 17 P← Merge(P,R)
                    offsprings = offsprings[:self.population_length - len(
                        self.population)]
                    self.population.extend(offsprings)

                    # 18 Sort(P,Rank,CD)
                    self.population, fronts = self.fast_nondominated_sort(
                        self.population)
                    for front in fronts:
                        self.calculate_crowding_distance(front)

                # 19 If size(P>pop size)
                elif len(self.population) > self.population_length:
                    # 20 truncate(P,size(P)-pop size)
                    self.population = self.truncate_population(fronts)

                self.num_generations += 1

                # SSS should be applied here and also in each front computed after extending the popoluation at
                # start of each iteration. it makes no sense to change the nsgaii fast filtering
                # if self.sss_per_iteration:
                # self.population = evaluation.solution_subset_selection.search_solution_subset(self.sss_type,
                # self.subset_size, self.population)

                self.best_generation, self.best_generation_avgValue = self.calculate_last_generation_with_enhance(
                    self.best_generation, self.best_generation_avgValue, self.num_generations,
                    self.population)

                if self.debug_mode:
                    self.debug_data(nds_debug=fronts[0])

        except EvaluationLimit:
            pass

        end = time.time()

        return {
            # 22 Pareto Front=P
            "population": fronts[0],  # TODO or self.population, check step 22
            "time": end - start,
            "nds_update_time": -1, #not applicable in nsgaiipt,
            "numGenerations": self.num_generations,
            "bestGeneration": self.best_generation,
            "best_individual": self.best_individual,
            "numEvaluations": self.num_evaluations,
            "nds_debug": self.nds_debug,
            "population_debug": self.population_debug
        }

    def selection_tournament(self, population: List[Solution]) -> List[Solution]:
        """Specific implementation of tournament selection that compares two solutions by the crowding operator
        """
        new_population = []
        for i in range(0, len(population)):
            best_candidate = None
            for j in range(0, self.selection_candidates):
                random_index = random.randint(0, len(population) - 1)
                candidate = population[random_index]

                if (best_candidate is None or self.crowding_operator(candidate, best_candidate) == 1):
                    best_candidate = candidate

            rank, crowding_distance = best_candidate.rank, best_candidate.crowding_distance
            best_candidate = Solution(
                self.dataset, None, selected=np.where(best_candidate.selected == 1))
            best_candidate.rank = rank
            best_candidate.crowding_distance = crowding_distance
            new_population.append(best_candidate)

        return new_population

    def fast_nondominated_sort(self, population: List[Solution]) -> Tuple[List[Solution], List[Solution]]:
        """Fast method that sorts a population in fronts, where each front contains solutions that are nondominated between them.
        """
        fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                # if not individual.__eq__(other_individual):##########################################
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                fronts[0].append(individual)
        i = 0
        while len(fronts[i]) > 0:
            temp = []
            for individual in fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            fronts.append(temp)
        return population, fronts

    def calculate_crowding_distance(self, front: List[Solution]) -> List[Solution]:
        """Calculates the crowding distance of each solution in the front given.
        """
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            front.sort(
                key=lambda individual: individual.total_cost)
            # front[0].crowding_distance = 10 ** 9 #########################
            front[0].crowding_distance = float('inf')
            # front[solutions_num - 1].crowding_distance = 10 ** 9 #########################
            front[solutions_num - 1].crowding_distance = float('inf')
            m_values = [
                individual.total_cost for individual in front]
            scale = max(m_values) - min(
                m_values)  # aqui calcula la escala o diferencia entre el valor mayor y el menor, y la usa para dividir en el crowding distance
            if scale == 0:
                scale = 1
            for i in range(1, solutions_num - 1):
                front[i].crowding_distance += (front[i + 1].total_cost -
                                               front[i - 1].total_cost) / scale

            front.sort(
                key=lambda individual: individual.total_satisfaction)
            # front[0].crowding_distance = 10 ** 9 #########################
            front[0].crowding_distance = float('inf')
            # front[solutions_num - 1].crowding_distance = 10 ** 9 #########################
            front[solutions_num - 1].crowding_distance = float('inf')
            m_values = [
                individual.total_satisfaction for individual in front]
            scale = max(m_values) - min(
                m_values)
            if scale == 0:
                scale = 1
            for i in range(1, solutions_num - 1):
                front[i].crowding_distance += (
                    front[i + 1].total_satisfaction - front[
                        i - 1].total_satisfaction) / scale

            return front

    def crowding_operator(self, individual: any, other_individual: any) -> int:
        """Compares two individuals by their dominance and/or crowding distance.
        """
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                    individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def remove_duplicates(self, population: List[Solution]) -> List[Solution]:
        """Removes duplicates from a population.
        """
        selected_dict = {}
        for individual in population:
            selected_str = str(individual.selected)
            selected_dict[selected_str] = individual

        return list(selected_dict.values())

    def truncate_population(self, fronts: List[List[Solution]]) -> List[Solution]:
        """Truncates the population to the maximum size allowed.
        """
        population = []
        front_num = 0
        # till parent population is filled, calculate crowding distance in Fi, include i-th non-dominated front in parent pop
        while len(population) + len(fronts[front_num]) <= self.population_length:
            self.calculate_crowding_distance(  # TODO already set previously, check
                fronts[front_num])
            population.extend(fronts[front_num])
            front_num += 1

        # ordenar los individuos del ultimo front por crowding distance y agregar los X que falten para completar la poblacion
        # TODO already set previously, check
        self.calculate_crowding_distance(fronts[front_num])

        # sort in descending order using >=n
        fronts[front_num].sort(
            key=lambda individual: individual.crowding_distance, reverse=True)

        # choose first N elements of Pt+1
        population.extend(
            fronts[front_num][0:self.population_length - len(population)])

        return population
