from algorithms.genetic.abstract_genetic.genetic_executer import GeneticExecuter


class GeneticNDSExecuter(GeneticExecuter):
    def __init__(self, algorithm):
        from algorithms.genetic.geneticnds.geneticnds_algorithm import GeneticNDSAlgorithm
        super().__init__(algorithm)
        self.algorithm: GeneticNDSAlgorithm
        self.algorithm_type = "genetic_nds"
