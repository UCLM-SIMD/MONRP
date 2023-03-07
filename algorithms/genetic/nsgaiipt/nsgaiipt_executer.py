from algorithms.genetic.abstract_genetic.genetic_executer import GeneticExecuter


class NSGAIIPTExecuter(GeneticExecuter):
    def __init__(self, algorithm, execs):
        from algorithms.genetic.nsgaiipt.nsgaiipt_algorithm import NSGAIIPTAlgorithm
        super().__init__(algorithm,execs)
        self.algorithm: NSGAIIPTAlgorithm
        self.algorithm_type = "nsgaiipt"
