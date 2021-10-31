from typing import Any, Dict, List
from algorithms.abstract_algorithm.abstract_executer import AbstractExecuter


class UMDAExecuter(AbstractExecuter):
    def __init__(self, algorithm):
        from algorithms.EDA.UMDA.umda_algorithm import UMDAAlgorithm
        super().__init__(algorithm)
        self.algorithm: UMDAAlgorithm
        self.algorithm_type: str = "umda"

        self.config_fields.extend(["Population Length", "MaxGenerations", "MaxEvaluations",
                                   "Selected Individuals"])

        self.metrics_fields.extend(
            ["NumGenerations", "NumEvaluations", ])

    def get_config_fields(self,) -> List[str]:
        config_lines: List[str] = super().get_config_fields()

        population_length = self.algorithm.solutions_per_iteration
        generations = self.algorithm.iterations
        max_evaluations = self.algorithm.max_evaluations
        selected_individuals = self.algorithm.selected_individuals

        config_lines.append(str(population_length))
        config_lines.append(str(generations))
        config_lines.append(str(max_evaluations))
        config_lines.append(str(selected_individuals))
        return config_lines

    def get_metrics_fields(self, result: Dict[str, Any]) -> List[str]:
        metrics_fields: List[str] = super().get_metrics_fields(result)

        numGenerations = str(
            result["numGenerations"]) if "numGenerations" in result else 'NaN'
        numEvaluations = str(
            result["numEvaluations"]) if "numEvaluations" in result else 'NaN'

        metrics_fields.append(str(numGenerations))
        metrics_fields.append(str(numEvaluations))

        return metrics_fields
