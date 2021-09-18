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
        #self.problem, self.dataset = self.utils.generate_dataset_problem(
        #    dataset_name=dataset_name)

        self.dataset = Dataset(dataset_name)
        self.dataset_name = dataset_name

        self.population_length = population_length
        self.max_generations = max_generations
        self.max_evaluations = max_evaluations

        self.selected_individuals=selected_individuals

        self.NDS=[]

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
            probs = np.full(
                self.dataset.pbis_score.size, 1/self.dataset.pbis_score.size)
            ind = GraspSolution(probs, costs=self.dataset.pbis_cost_scaled,
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

    def calculate_probabilities(self, population):
        probability_model = []
        for index in np.arange(len(self.dataset.pbis_cost_scaled)):
            num_ones = 0
            for individual in population:
                num_ones += individual.selected[index]
            index_probability = num_ones/len(population)
            probability_model.append(index_probability)

        return probability_model

    def generate_sample_from_probabilities(self, probabilities):
        sample_selected = np.random.binomial(1, probabilities)
        sample=GraspSolution(None, costs=self.dataset.pbis_cost_scaled,
                                           values=self.dataset.pbis_satisfaction_scaled,selected=sample_selected)
        return sample

    def replace_population_from_probabilities(self, probability_model, population):
        new_population = []

        # elitist R-1 inds
        for i in np.arange(self.population_length-1):
            #new_individual = GraspSolution(probability_model, costs=self.dataset.pbis_cost_scaled,
            #                               values=self.dataset.pbis_satisfaction_scaled)
            #new_individual= np.random.choice([0, 1], size=len(self.dataset.pbis_cost_scaled), p=probability_model)
            new_individual = self.generate_sample_from_probabilities(probability_model)
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
        genes,_ = generate_dataset_genes(self.dataset.id)
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

    def reset(self):
        self.NDS = []
        self.best_individual = None

    def stop_criterion(self, num_generations, num_evaluations):
        if self.max_evaluations is 0:
            return num_generations < self.max_generations
        else:
            return num_evaluations < self.max_evaluations
        
    # RUN ALGORITHM------------------------------------------------------------------
    def run(self):
        self.reset()
        start = time.time()

        num_generations = 0
        num_evaluations = 0

        returned_population = None
        self.population = self.generate_starting_population()
        #print("STARTING")
        #for i in self.population:
        #    print(i)
        self.evaluate(self.population, self.best_individual)

        #while num_generations < self.max_generations:
        while (self.stop_criterion(num_generations, num_evaluations)):
            #print("--------------------------------")
            #("SELECT")
            individuals = self.select_individuals(self.population)
            #for i in individuals:
            #    print(i)
            probability_model = self.calculate_probabilities(individuals)
            #print("PROB MODEL")
            #print(probability_model)
            self.population = self.replace_population_from_probabilities(
                probability_model, self.population)
            #print("REPLACEMENT")
            #for i in self.population:
            #    print(i)
            self.evaluate(self.population, self.best_individual)
            # update NDS with solutions constructed and evolved in this iteration
            self.update_nds(self.population)
            
            num_generations += 1

        self.format()
        end = time.time()

        return {"population": self.NDS,  
                "time": end - start,
                "numGenerations": num_generations,
                "best_individual": self.best_individual,
                "numEvaluations": num_evaluations
                }
