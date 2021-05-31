from algorithms.abstract_genetic.basegenetic_executer import BaseGeneticExecuter

class NSGAIIExecuter(BaseGeneticExecuter):
    def __init__(self,algorithm):
        super().__init__(algorithm)
        self.algorithm_type = "nsgaii"