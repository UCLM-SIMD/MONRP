from algorithms.abstract_default.evaluation_exception import EvaluationLimit
from models.solution import Solution
from models.problem import Problem
from datasets.dataset_gen_generator import generate_dataset_genes
from algorithms.GRASP.GraspSolution import GraspSolution
from algorithms.GRASP.Dataset import Dataset
from algorithms.EDA.UMDA.umda_executer import UMDAExecuter
from algorithms.genetic.abstract_genetic.basegenetic_algorithm import BaseGeneticAlgorithm
from algorithms.genetic.genetic.genetic_executer import GeneticExecuter
from algorithms.genetic.genetic.genetic_utils import GeneticUtils

import copy
import time
import numpy as np


class UMDAAlgorithm():  # Univariate Marginal Distribution Algorithm
    def __init__(self, dataset_name="1", random_seed=None, population_length=100, max_generations=100, max_evaluations=0,
                 selected_individuals=60):

        self.executer = UMDAExecuter(algorithm=self)
        # self.problem, self.dataset = self.utils.generate_dataset_problem(
        #    dataset_name=dataset_name)

        self.dataset = Dataset(dataset_name)
        self.dataset_name = dataset_name

        self.population_length = population_length
        self.max_generations = max_generations
        self.max_evaluations = max_evaluations

        self.selected_individuals = selected_individuals

        self.nds = []
        self.num_evaluations = 0
        self.num_generations = 0
        self.best_individual = None

        # TODO los utils no se usan y estan mal los super()

        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.file = str(self.__class__.__name__)+"-"+str(dataset_name)+"-"+str(random_seed)+"-"+str(population_length)+"-" +\
            str(max_generations)+".txt"

    def get_name(self):
        return "UMDA"

    def generate_starting_population(self):
        population = []
        for i in np.arange(self.population_length):
            candidates_score_scaled = self.dataset.pbis_score / self.dataset.pbis_score.sum()

            # np.random.uniform(low=0, high=1, size=num_vars)


            # probs = np.full(
            # self.dataset.pbis_score.size, 1/self.dataset.pbis_score.size)  # 0.5 ?
            # ind = GraspSolution(probs, costs=self.dataset.pbis_cost_scaled,
            #                    values=self.dataset.pbis_satisfaction_scaled)
            ind = GraspSolution(candidates_score_scaled, costs=self.dataset.pbis_cost_scaled,
                                values=self.dataset.pbis_satisfaction_scaled)
            population.append(ind)

        return population

    def select_individuals(self, population):
        individuals = []
        population.sort(
            key=lambda x: x.compute_mono_objective_score(), reverse=True)

        for i in np.arange(self.selected_individuals):
            individuals.append(population[i])

        return individuals

    def select_nondominated_individuals(self, population):
        selected_individuals = self.update_nds(population, [])
        return selected_individuals

    def calculate_probabilities(self, population):
        probability_model = []
        # para cada gen:
        for index in np.arange(len(self.dataset.pbis_cost_scaled)):
            num_ones = 0
            # contar cuantos 1 hay en la poblacion
            for individual in population:
                num_ones += individual.selected[index]
            # prob = nº 1s / nº total
            index_probability = num_ones/len(population)
            probability_model.append(index_probability)

        return probability_model

    def generate_sample_from_probabilities(self, probabilities):
        sample_selected = np.random.binomial(1, probabilities)
        sample = GraspSolution(None, costs=self.dataset.pbis_cost_scaled,
                               values=self.dataset.pbis_satisfaction_scaled, selected=sample_selected)
        return sample

    def replace_population_from_probabilities(self, probability_model, population):
        new_population = []

        # elitist R-1 inds
        for i in np.arange(self.population_length-1):
            # new_individual = GraspSolution(probability_model, costs=self.dataset.pbis_cost_scaled,
            #                               values=self.dataset.pbis_satisfaction_scaled)
            #new_individual= np.random.choice([0, 1], size=len(self.dataset.pbis_cost_scaled), p=probability_model)
            new_individual = self.generate_sample_from_probabilities(
                probability_model)
            new_population.append(new_individual)

        # elitism -> add best individual from old population
        population.sort(
            key=lambda x: x.compute_mono_objective_score(), reverse=True)
        new_population.append(population[0])

        return new_population

    def evaluate(self, population, best_individual):
        best_score = 0
        new_best_individual = None
        for ind in population:
            if ind.compute_mono_objective_score() > best_score:
                new_best_individual = copy.deepcopy(ind)
                best_score = ind.compute_mono_objective_score()
            self.add_evaluation(population)
        if best_individual is not None:
            if new_best_individual.compute_mono_objective_score() > best_individual.compute_mono_objective_score():
                best_individual = copy.deepcopy(new_best_individual)
        else:
            best_individual = copy.deepcopy(new_best_individual)

    def add_evaluation(self, new_population):
        self.num_evaluations += 1
        # if(self.num_evaluations >= self.max_evaluations):
        if (self.stop_criterion(self.num_generations, self.num_evaluations)):
            self.update_nds(new_population, self.nds)
            raise EvaluationLimit

    def update_nds(self, solutions, nds):
        """
        For each sol in solutions:
            if no solution in nds dominates sol:
             insert sol in nds
             remove all solutions in self.nds now dominated by sol
        :param solutions: solutions created in current GRASP iteration and evolved with local search
        """
        for sol in solutions:
            insert = True

            # find which solutions, if any, in self.nds are dominated by sol
            # if sol is dominated by any solution in self.nds, then search is stopped and sol is discarded
            now_dominated = []
            for nds_sol in nds:
                if np.array_equal(sol.selected, nds_sol.selected):
                    insert = False
                    break
                else:
                    if sol.dominates(nds_sol):
                        now_dominated.append(nds_sol)
                    # do not insert if sol is dominated by a solution in self.nds
                    if nds_sol.dominates(sol):
                        insert = False
                        break

            # sol is inserted if it is not dominated by any solution in self.nds,
            # then all solutions in self.nds dominated by sol are removed
            if insert:
                nds.append(sol)
                for dominated in now_dominated:
                    nds.remove(dominated)

        return nds

    def format(self,population):
        genes, _ = generate_dataset_genes(self.dataset.id)
        problem = Problem(genes, ["MAX", "MIN"])
        final_nds_formatted = []

        for solution in population:
            # print(solution)
            individual = Solution(problem.genes, problem.objectives)
            for b in np.arange(len(individual.genes)):
                individual.genes[b].included = solution.selected[b]
            individual.evaluate_fitness()
            final_nds_formatted.append(individual)
        population = final_nds_formatted
        return population

    def reset(self):
        self.nds = []
        self.best_individual = None
        self.num_generations = 0
        self.num_evaluations = 0

    def stop_criterion(self, num_generations, num_evaluations):
        if self.max_evaluations == 0:
            return num_generations >= self.max_generations
        else:
            return num_evaluations >= self.max_evaluations

    # RUN ALGORITHM------------------------------------------------------------------
    def run(self):
        self.reset()
        start = time.time()

        returned_population = None
        self.population = self.generate_starting_population()
        # print("STARTING")
        # for i in self.population:
        #    print(i)
        self.evaluate(self.population, self.best_individual)

        # while num_generations < self.max_generations:
        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                # print("--------------------------------")
                # ("SELECT")
                #individuals = self.select_individuals(self.population)
                individuals = self.select_nondominated_individuals(
                    self.population)

                # for i in individuals:
                #    print(i)
                probability_model = self.calculate_probabilities(
                    individuals)  # joint probability esta bien? TODO
                #print("PROB MODEL")
                # print(probability_model)
                self.population = self.replace_population_from_probabilities(
                    probability_model, self.population)
                # print("REPLACEMENT")
                # for i in self.population:
                #    print(i)
                self.evaluate(self.population, self.best_individual)

                # update nds with solutions constructed and evolved in this iteration
                self.update_nds(self.population, self.nds)

                self.num_generations += 1

        except EvaluationLimit:
            pass

        self.nds=self.format(self.nds)    
        end = time.time()

        print("\nNDS created has", self.nds.__len__(), "solution(s)")

        return {"population": self.nds,
                "time": end - start,
                "numGenerations": self.num_generations,
                "best_individual": self.best_individual,
                "numEvaluations": self.num_evaluations
                }
