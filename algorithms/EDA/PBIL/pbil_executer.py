from typing import Any, Dict, List
from algorithms.abstract_algorithm.abstract_executer import AbstractExecuter


class PBILExecuter(AbstractExecuter):
    """Specific pbil implementation of executer.
    """

    def __init__(self, algorithm,execs:int):
        """Init method extends config and metrics fields with specific pbil algorithm data
        """
        from algorithms.EDA.PBIL.pbil_algorithm import PBILAlgorithm
        super().__init__(algorithm,execs)
        self.algorithm: PBILAlgorithm
        self.algorithm_type: str = "pbil"

        self.config_fields.extend(["Population Length", "MaxGenerations", "MaxEvaluations",
                                   "Learning Rate", "Mutation Probability", "Mutation Shift"])

        self.metrics_fields.extend(
            ["NumGenerations", "NumEvaluations", ])
        self.metrics_dictionary["NumGenerations"] = [None] * int(execs)
        self.metrics_dictionary["NumEvaluations"] = [None] * int(execs)

    def get_config_fields(self,) -> List[str]:
        """PBIL algorithm executer extends config fields read from the execution
        """
        config_lines: List[str] = super().get_config_fields()

        population_length = self.algorithm.population_length
        max_generations = self.algorithm.max_generations
        max_evaluations = self.algorithm.max_evaluations
        learning_rate = self.algorithm.learning_rate
        mutation_prob = self.algorithm.mutation_prob
        mutation_shift = self.algorithm.mutation_shift

        config_lines.append(str(population_length))
        config_lines.append(str(max_generations))
        config_lines.append(str(max_evaluations))
        config_lines.append(str(learning_rate))
        config_lines.append(str(mutation_prob))
        config_lines.append(str(mutation_shift))
        return config_lines

    def get_metrics_fields(self, result: Dict[str, Any], repetition) -> List[str]:
        """PBIL algorithm executer extends metrics fields read from the execution
        """
        metrics_fields: List[str] = super().get_metrics_fields(result, repetition)

        numGenerations = result["numGenerations"] if "numGenerations" in result else 'NaN'
        numEvaluations = result["numEvaluations"] if "numEvaluations" in result else 'NaN'

        self.metrics_dictionary['NumGenerations'][repetition] = numGenerations
        self.metrics_dictionary['NumEvaluations'][repetition] = numEvaluations

        return metrics_fields
