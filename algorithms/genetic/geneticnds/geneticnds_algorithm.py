from typing import Any, Dict, List
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit
from algorithms.genetic.abstract_genetic.abstract_genetic_algorithm import AbstractGeneticAlgorithm
from algorithms.genetic.geneticnds.geneticnds_executer import GeneticNDSExecuter
import copy
import time
from evaluation.get_nondominated_solutions import get_nondominated_solutions
import random
from models.Solution import Solution


class GeneticNDSAlgorithm(AbstractGeneticAlgorithm):
    """Mono-Objective Genetic Algorithm that stores a set of nondominated solutions and updates it at each generation.
    """

    def __init__(self, dataset_name: str = "test", random_seed: int = None, debug_mode: bool = False, tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0,
                 selection: str = "tournament", selection_candidates: int = 2,
                 crossover: str = "onepoint", crossover_prob: float = 0.9,
                 mutation: str = "flipeachbit", mutation_prob: float = 0.1,
                 replacement: str = "elitism",):

        super().__init__(dataset_name, random_seed, debug_mode, tackle_dependencies,
                         population_length, max_generations, max_evaluations,
                         selection, selection_candidates, crossover, crossover_prob,
                         mutation, mutation_prob, replacement,)

        self.executer = GeneticNDSExecuter(algorithm=self)

        if selection == "tournament":
            self.selection = self.selection_tournament

        if crossover == "onepoint":
            self.crossover = self.crossover_one_point

        if mutation == "flip1bit":
            self.mutation = self.mutation_flip1bit
        elif mutation == "flipeachbit":
            self.mutation = self.mutation_flipeachbit

        if replacement == "elitism":
            self.replacement = self.replacement_elitism
        elif replacement == "elitismnds":
            self.replacement = self.replacement_elitism

        self.file: str = (f"{str(self.__class__.__name__)}-{str(dataset_name)}-{str(random_seed)}-{str(population_length)}-"
                          f"{str(max_generations)}-{str(max_evaluations)}-"
                          f"{str(selection)}-{str(selection_candidates)}-"
                          f"{str(crossover)}-{str(crossover_prob)}-{str(mutation)}-"
                          f"{str(mutation_prob)}-{str(replacement)}.txt")

    def get_name(self) -> str:
        return f"GeneticNDS{str(self.population_length)}+{str(self.max_generations)}+{str(self.max_evaluations)}+{str(self.crossover_prob)}\
            +{str(self.mutation_scheme)}+{str(self.mutation_prob)}"

    def reset(self) -> None:
        """Specific reset implementation
        """
        super().reset()
        self.nds = []

    # RUN ALGORITHM------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        self.reset()
        paretos = []
        start = time.time()

        self.num_generations = 0
        self.num_evaluations = 0
        self.population = self.generate_starting_population()
        self.evaluate(self.population, self.best_individual)

        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                # selection
                new_population = self.selection(self.population)
                # crossover
                new_population = self.crossover(new_population)

                # mutation
                new_population = self.mutation(new_population)

                # repair population if dependencies tackled:
                if(self.tackle_dependencies):
                    new_population = self.repair_population_dependencies(
                        new_population)

                # evaluation
                self.evaluate(self.population, self.best_individual)

                # update NDS
                get_nondominated_solutions(new_population, self.nds)

                returned_population = copy.deepcopy(new_population)
                self.best_generation, self.best_generation_avgValue = self.calculate_last_generation_with_enhance(
                    self.best_generation, self.best_generation_avgValue, self.num_generations, returned_population)

                # replacement
                if self.replacement_scheme == "elitismnds":
                    self.population = self.replacement(
                        self.nds, new_population)
                else:
                    self.population = self.replacement(
                        self.population, new_population)

                self.num_generations += 1
                if self.debug_mode:
                    paretos.append(self.nds)

        except EvaluationLimit:
            pass

        end = time.time()

        return {
            "population": self.nds,
            "time": end - start,
            "numGenerations": self.num_generations,
            "bestGeneration": self.best_generation,
            "best_individual": self.best_individual,
            "numEvaluations": self.num_evaluations,
            "paretos": paretos,
        }

    def add_evaluation(self, new_population: List[Solution]) -> None:
        """Handles evaluation count, finishing algorithm execution if stop criterion is met by raising an exception.
        """
        self.num_evaluations += 1
        if (self.stop_criterion(self.num_generations, self.num_evaluations)):
            get_nondominated_solutions(new_population, self.nds)
            raise EvaluationLimit

    def selection_tournament(self, population: List[Solution]) -> List[Solution]:
        """Each time, chooses randomly a number of candidates, seleting the one with highest monoscore value.
        """
        new_population = []
        # create as many individuals as pop. size
        for i in range(0, len(population)):
            best_candidate = None
            best_total_score = 0
            # select individual from a set of X candidates
            for j in range(0, self.selection_candidates):
                random_index = random.randint(0, len(population)-1)
                candidate = population[random_index]

                score = candidate.total_satisfaction
                cost = candidate.total_cost
                total_score = candidate.mono_objective_score
                # store best scoring individual
                if(total_score > best_total_score):
                    best_total_score = total_score
                    best_candidate = candidate

            # append individual in new population
            new_population.append(best_candidate)

        return new_population

    def replacement_elitism(self, population: List[Solution], newpopulation: List[Solution]) -> List[Solution]:
        """Replacement method that keeps the best individual of the former population into the new one.
        """
        # find best individual in former population
        best_individual = None
        best_individual_total_score = 0
        for ind in population:
            if (ind.mono_objective_score > best_individual_total_score):
                best_individual_total_score = ind.mono_objective_score
                best_individual = ind

        # find index of worst individual in new population
        newpopulation_replaced = []
        newpopulation_replaced.extend(newpopulation)

        worst_individual_total_score = float('inf')
        worst_individual_index = None
        for ind in newpopulation_replaced:
            if (ind.mono_objective_score < worst_individual_total_score):
                worst_individual_total_score = ind.mono_objective_score
                worst_individual_index = newpopulation_replaced.index(ind)

        # replace worst individual by best one
        newpopulation_replaced[worst_individual_index] = best_individual
        return newpopulation_replaced
