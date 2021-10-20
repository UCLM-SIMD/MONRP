from algorithms.abstract_default.evaluation_exception import EvaluationLimit
from algorithms.abstract_default.algorithm import Algorithm


class BaseGeneticAlgorithm(Algorithm):
    def __init__(self):
        pass

    def reset(self):
        pass

    def run(self):
        pass
    
    def get_name(self):
        pass

    def stop_criterion(self, num_generations, num_evaluations):
        if self.max_evaluations == 0:
            return num_generations >= self.max_generations
        else:
            return num_evaluations >= self.max_evaluations