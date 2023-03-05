from typing import Any, Dict, List
from algorithms.abstract_algorithm.abstract_executer import AbstractExecuter


class RandomExecuter(AbstractExecuter):
    """Specific random implementation of executer.
    """

    def __init__(self, algorithm, execs: int):
        """Init method extends config and metrics fields with specific genetic algorithm data
        """
        from algorithms.random.random_algorithm import RandomAlgorithm
        super().__init__(algorithm, execs)
        self.algorithm: RandomAlgorithm
        self.algorithm_type: str = "random"

        self.config_fields.extend(
            ["Population Length", "MaxGenerations", "MaxEvaluations"])

        self.metrics_fields.extend(
            ["NumGenerations", "NumEvaluations"])
        self.metrics_dictionary["NumGenerations"] = [None] * int(execs)
        self.metrics_dictionary["NumEvaluations"] = [None] * int(execs)

    def get_config_fields(self) -> List[str]:
        """Genetic algorithm executer extends config fields read from the execution
        """
        config_lines: List[str] = super().get_config_fields()

        population_length = self.algorithm.population_length
        generations = self.algorithm.max_generations
        max_evaluations = self.algorithm.max_evaluations

        config_lines.append(str(population_length))
        config_lines.append(str(generations))
        config_lines.append(str(max_evaluations))
        return config_lines

    def get_metrics_fields(self, result: Dict[str, Any], repetition) -> List[str]:
        """Genetic algorithm executer extends metrics fields read from the execution
        """
        metrics_fields: List[str] = super(
        ).get_metrics_fields(result, repetition)

        numGenerations = result["numGenerations"] if "numGenerations" in result else 'NaN'
        numEvaluations = result["numEvaluations"] if "numEvaluations" in result else 'NaN'

        self.metrics_dictionary['NumGenerations'][repetition] = numGenerations
        self.metrics_dictionary['NumEvaluations'][repetition] = numEvaluations

        return metrics_fields
