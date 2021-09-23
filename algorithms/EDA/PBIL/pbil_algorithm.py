from algorithms.abstract_default.evaluation_exception import EvaluationLimit
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
    def __init__(self, dataset_name="1", random_seed=None, population_length=20, max_generations=100, max_evaluations=0,
                 learning_rate=0.1, mutation_prob=0.1, mutation_shift=0.1):

        self.executer = PBILExecuter(algorithm=self)
        # self.problem, self.dataset = self.utils.generate_dataset_problem(
        #    dataset_name=dataset_name)

        self.dataset = Dataset(dataset_name)
        self.dataset_name = dataset_name

        self.population_length = population_length
        self.max_generations = max_generations
        self.max_evaluations = max_evaluations

        self.learning_rate = learning_rate
        self.mutation_prob = mutation_prob
        self.mutation_shift = mutation_shift

        self.nds = []
        self.best_individual = None
        self.num_generations = 0
        self.num_evaluations = 0

        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.file = str(self.__class__.__name__)+"-"+str(dataset_name)+"-"+str(random_seed)+"-"+str(population_length)+"-" +\
            str(max_generations)+ "-"+str(max_evaluations)+"-"+str(learning_rate)+"-" + \
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
            self.add_evaluation(population)
        if best_individual is not None:
            if new_best_individual.compute_mono_objective_score() > best_individual.compute_mono_objective_score():
                best_individual = copy.deepcopy(new_best_individual)
        else:
            best_individual = copy.deepcopy(new_best_individual)

    def add_evaluation(self,new_population):
        self.num_evaluations+=1
        #if(self.num_evaluations >= self.max_evaluations):
        if (self.stop_criterion(self.num_generations, self.num_evaluations)):
            self.update_nds(new_population)
            raise EvaluationLimit

    def update_nds(self, solutions):
        """
        For each sol in solutions:
            if no solution in self.nds dominates sol:
             insert sol in self.nds
             remove all solutions in self.nds now dominated by sol
        :param solutions: solutions created in current GRASP iteration and evolved with local search
        """
        for sol in solutions:
            insert = True

            # find which solutions, if any, in self.nds are dominated by sol
            # if sol is dominated by any solution in self.nds, then search is stopped and sol is discarded
            now_dominated = []
            for nds_sol in self.nds:
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
                self.nds.append(sol)
                for dominated in now_dominated:
                    self.nds.remove(dominated)

    def format(self, population):
        genes, _ = generate_dataset_genes(self.dataset.id)
        problem = Problem(genes, ["MAX", "MIN"])
        final_nds_formatted = []

        for solution in population:
            #print(solution)
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
        start = time.time()
        self.reset()


        returned_population = None
        self.probability_vector = self.initialize_probability_vector()

        #while num_generations < self.max_generations:
        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                self.population = []
                for i in np.arange(self.population_length):
                    sample = self.generate_sample_from_probabilities(
                        self.probability_vector)
                    self.population.append(sample)

                self.evaluate(self.population, self.best_individual)

                max_value, max_sample = self.find_max_sample(self.population) # esto es muy monobjetivo

                for i in np.arange(len(self.probability_vector)):
                    self.probability_vector[i] = self.probability_vector[i]*(
                        1-self.learning_rate)+max_sample.selected[i]*(self.learning_rate)

                for i in np.arange(len(self.probability_vector)): # mutacion flip each bit
                    prob = np.random.random_sample()
                    if prob < self.mutation_prob:
                        self.probability_vector[i] = self.probability_vector[i]*(
                            1-self.mutation_shift) + (np.random.randint(2))*self.mutation_shift

                # update nds with solutions constructed and evolved in this iteration
                self.update_nds(self.population)
                self.num_generations += 1

        except EvaluationLimit:
            pass

        self.nds = self.format(self.nds)
        end = time.time()

        print("\nNDS created has", self.nds.__len__(), "solution(s)")

        return {"population": self.nds,
                "time": end - start,
                "numGenerations": self.num_generations,
                "best_individual": self.best_individual,
                "numEvaluations": self.num_evaluations
                }
