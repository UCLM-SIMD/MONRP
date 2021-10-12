import copy

import numpy as np
from algorithms.GRASP.GraspSolution import GraspSolution
from algorithms.abstract_default.algorithm import Algorithm
from algorithms.abstract_default.evaluation_exception import EvaluationLimit
from evaluation.update_nds import get_nondominated_solutions


class EDAAlgorithm(Algorithm):  # Estimation of Distribution Algorithm
    ''' 
    GENERATE INITIAL POPULATION
    '''

    def generate_initial_population(self):
        population = []
        candidates_score_scaled = np.full(
            self.dataset.pbis_score.size, 1/self.dataset.pbis_score.size)
        for i in np.arange(self.population_length):
            #candidates_score_scaled = self.dataset.pbis_score / self.dataset.pbis_score.sum()

            # probs = np.full(
            # self.dataset.pbis_score.size, 1/self.dataset.pbis_score.size)  # 0.5 ?
            # ind = GraspSolution(probs, costs=self.dataset.pbis_cost_scaled,
            #                    values=self.dataset.pbis_satisfaction_scaled)

            ind = GraspSolution(candidates_score_scaled, costs=self.dataset.pbis_cost_scaled,
                                values=self.dataset.pbis_satisfaction_scaled)

            # ind = GraspSolution(None, costs=self.dataset.pbis_cost_scaled,
            #                    values=self.dataset.pbis_satisfaction_scaled,uniform=True)

            population.append(ind)
        return population

    ''' 
    SELECT INDIVIDUALS
    '''

    def select_individuals(self, population):
        if self.selection_scheme == "nds":
            individuals = self.select_nondominated_individuals(
                population)
        elif self.selection_scheme == "monoscore":
            individuals = self.select_individuals_monoscore(population)
        return individuals

    def select_individuals_monoscore(self, population):
        individuals = []
        population.sort(
            key=lambda x: x.compute_mono_objective_score(), reverse=True)

        for i in np.arange(self.selected_individuals):
            individuals.append(population[i])

        return individuals

    def select_nondominated_individuals(self, population):
        selected_individuals = get_nondominated_solutions(population, [])
        return selected_individuals

    ''' 
    LEARN PROBABILITY MODEL
    '''

    def learn_probability_model(self):
        pass

    ''' 
    SAMPLE NEW POPULATION
    '''

    def sample_new_population(self):
        pass

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
            self.update_nds(new_population)
            raise EvaluationLimit

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
