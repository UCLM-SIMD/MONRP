from algorithms.abstract_default.evaluation_exception import EvaluationLimit
from algorithms.abstract_default.algorithm import Algorithm


class BaseGeneticAlgorithm(Algorithm):
    def __init__(self, dataset_name: str = "test", random_seed: int = None, debug_mode: bool = False, tackle_dependencies: bool = False,
                 population_length: int = 100, max_generations: int = 100, max_evaluations: int = 0,
                 selection:str="tournament", selection_candidates:int=2, crossover:str="onepoint", crossover_prob:float=0.9,
                 mutation:str="flipeachbit", mutation_prob:float=0.1,
                 replacement:str="elitism",):

        super().__init__(dataset_name, random_seed, debug_mode, tackle_dependencies)

        self.population_length: int = population_length
        self.max_generations: int = max_generations
        self.max_evaluations: int = max_evaluations

        self.selection_scheme:str = selection
        self.selection_candidates:int = selection_candidates
        self.crossover_scheme:str = crossover
        self.crossover_prob:float = crossover_prob
        self.mutation_scheme:str = mutation
        self.mutation_prob:float = mutation_prob
        self.replacement_scheme:str = replacement

        self.population = None
        self.best_generation_avgValue = None
        self.best_generation = None

        self.nds = []
        self.num_evaluations: int = 0
        self.num_generations: int = 0
        self.best_individual = None

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
