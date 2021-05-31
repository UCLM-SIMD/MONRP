from algorithms.abstract_genetic.basegenetic_executer import BaseGeneticExecuter


class GeneticNDSExecuter(BaseGeneticExecuter):
    def __init__(self, algorithm):
        super().__init__(algorithm)
        self.algorithm_type = "genetic_nds"
