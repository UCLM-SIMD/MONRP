from algorithms.genetic.abstract_genetic.genetic_executer import GeneticExecuter


class AGEMOEAExecuter(GeneticExecuter):
    def __init__(self, algorithm, execs):
        from algorithms.genetic.agemoea2.agemoea2_algorithm import AGEMOEA2Algorithm
        super().__init__(algorithm,execs)
        self.algorithm: AGEMOEA2Algorithm
        self.algorithm_type = "AGE-MOEA2"
