from algorithms.genetic.abstract_genetic.genetic_executer import GeneticExecuter


class CTAEAExecuter(GeneticExecuter):
    def __init__(self, algorithm, execs):
        from algorithms.genetic.ctaea.ctaea_algorithm import CTAEAAlgorithm
        super().__init__(algorithm,execs)
        self.algorithm: CTAEAAlgorithm
        self.algorithm_type = "C-TAEA"
