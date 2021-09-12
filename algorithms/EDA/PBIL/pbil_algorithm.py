from models.solution import Solution
from datasets.dataset_gen_generator import generate_dataset_genes
from models.problem import Problem
from algorithms.EDA.PBIL.pbil_executer import PBILExecuter
from algorithms.GRASP.GraspSolution import GraspSolution
from algorithms.GRASP.Dataset import Dataset
from algorithms.genetic.abstract_genetic.basegenetic_algorithm import BaseGeneticAlgorithm
from algorithms.genetic.genetic.genetic_executer import GeneticExecuter
from algorithms.genetic.genetic.genetic_utils import GeneticUtils

import copy
import time
import numpy as np


class PBILAlgorithm():  # Population Based Incremental Learning
    def __init__(self, dataset_name="1", random_seed=None, population_length=20, max_generations=100,
                 learning_rate=0.1, mutation_prob=0.1, mutation_shift=0.1):

        self.executer = PBILExecuter(algorithm=self)
        # self.problem, self.dataset = self.utils.generate_dataset_problem(
        #    dataset_name=dataset_name)

        self.dataset = Dataset(dataset_name)
        self.dataset_name = dataset_name

        self.population_length = population_length
        self.max_generations = max_generations
        self.learning_rate = learning_rate
        self.mutation_prob = mutation_prob
        self.mutation_shift = mutation_shift

        self.NDS = []

        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.file = str(self.__class__.__name__)+"-"+str(dataset_name)+"-"+str(random_seed)+"-"+str(population_length)+"-" +\
            str(max_generations)+"-"+str(learning_rate)+"-" + \
            str(mutation_prob)+"-"+str(mutation_shift)+".txt"

    def get_name(self):
        return "PBIL"

    def initialize_probability_vector(self):
        probabilities = np.full(self.dataset.pbis_score.size, 0.5)
        return probabilities

    def generate_sample_from_probabilities(self, probabilities):
        sample_selected = np.random.binomial(1, probabilities)
        sample = GraspSolution(None, costs=self.dataset.pbis_cost_scaled,
                               values=self.dataset.pbis_satisfaction_scaled, selected=sample_selected)
        return sample

    def find_max_sample(self, population):
        population.sort(
            key=lambda x: x.compute_mono_objective_score(), reverse=True)
        max_score = population[0].compute_mono_objective_score()
        max_sample = population[0]

        return max_score, max_sample

    def evaluate(self, population, best_individual):
        best_score = 0
        new_best_individual = None
        for ind in population:
            if ind.compute_mono_objective_score() > best_score:
                new_best_individual = copy.deepcopy(ind)
                best_score = ind.compute_mono_objective_score()

        if best_individual is not None:
            if new_best_individual.compute_mono_objective_score() > best_individual.compute_mono_objective_score():
                best_individual = copy.deepcopy(new_best_individual)
        else:
            best_individual = copy.deepcopy(new_best_individual)

    def update_nds(self, solutions):
        """
        For each sol in solutions:
            if no solution in self.NDS dominates sol:
             insert sol in self.NDS
             remove all solutions in self.NDS now dominated by sol
        :param solutions: solutions created in current GRASP iteration and evolved with local search
        """
        for sol in solutions:
            insert = True

            # find which solutions, if any, in self.NDS are dominated by sol
            # if sol is dominated by any solution in self.NDS, then search is stopped and sol is discarded
            now_dominated = []
            for nds_sol in self.NDS:
                if np.array_equal(sol.selected, nds_sol.selected):
                    insert = False
                    break
                else:
                    if sol.dominates(nds_sol):
                        now_dominated.append(nds_sol)
                    # do not insert if sol is dominated by a solution in self.NDS
                    if nds_sol.dominates(sol):
                        insert = False
                        break

            # sol is inserted if it is not dominated by any solution in self.NDS,
            # then all solutions in self.NDS dominated by sol are removed
            if insert:
                self.NDS.append(sol)
                for dominated in now_dominated:
                    self.NDS.remove(dominated)

    def format(self):
        genes, _ = generate_dataset_genes(self.dataset.id)
        problem = Problem(genes, ["MAX", "MIN"])
        final_nds_formatted = []

        for solution in self.NDS:
            #print(solution)
            individual = Solution(problem.genes, problem.objectives)
            for b in np.arange(len(individual.genes)):
                individual.genes[b].included = solution.selected[b]
            individual.evaluate_fitness()
            final_nds_formatted.append(individual)
        self.NDS = final_nds_formatted

    # RUN ALGORITHM------------------------------------------------------------------
    def reset(self):
        self.NDS = []
        self.best_individual = None

    def run(self):
        start = time.time()
        self.reset()
        num_generations = 0
        returned_population = None
        self.best_individual = None
        self.probability_vector = self.initialize_probability_vector()

        while num_generations < self.max_generations:
            self.population = []
            for i in np.arange(self.population_length):
                sample = self.generate_sample_from_probabilities(
                    self.probability_vector)
                self.population.append(sample)

            self.evaluate(self.population, self.best_individual)

            max_value, max_sample = self.find_max_sample(self.population)

            for i in np.arange(len(self.probability_vector)):
                self.probability_vector[i] = self.probability_vector[i]*(
                    1-self.learning_rate)+max_sample.selected[i]*(self.learning_rate)

            for i in np.arange(len(self.probability_vector)):
                prob = np.random.random_sample()
                if prob < self.mutation_prob:
                    self.probability_vector[i] = self.probability_vector[i]*(
                        1-self.mutation_shift) + (np.random.randint(2))*self.mutation_shift

            # update NDS with solutions constructed and evolved in this iteration
            self.update_nds(self.population)
            num_generations += 1

        self.format()
        end = time.time()

        return {"population": self.NDS,
                "time": end - start,
                "numGenerations": num_generations,
                "best_individual": self.best_individual,
                }
