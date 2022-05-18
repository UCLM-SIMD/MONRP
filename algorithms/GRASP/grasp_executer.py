from typing import Any, Dict, List
from algorithms.abstract_algorithm.abstract_executer import AbstractExecuter


class GRASPExecuter(AbstractExecuter):
    def __init__(self, algorithm, execs: int):
        from algorithms.GRASP.GRASP import GRASP
        super().__init__(algorithm, execs)
        self.algorithm: GRASP
        self.algorithm_type: str = "grasp"

        self.config_fields.extend(["Population Length", "MaxGenerations", "MaxEvaluations",
                                   "Initialization Type", "Local Search Type", "Path Relinking"])

        self.metrics_fields.extend(
            ["NumGenerations", "NumEvaluations", ])
        self.metrics_dictionary["NumGenerations"] = [None] * int(execs)
        self.metrics_dictionary["NumEvaluations"] = [None] * int(execs)

    def get_config_fields(self,) -> List[str]:
        config_lines: List[str] = super().get_config_fields()

        population_length = self.algorithm.solutions_per_iteration
        generations = self.algorithm.iterations
        max_evaluations = self.algorithm.max_evaluations
        init_type = self.algorithm.init_type
        local_search_type = self.algorithm.local_search_type
        path_relinking_mode = self.algorithm.path_relinking_mode

        config_lines.append(str(population_length))
        config_lines.append(str(generations))
        config_lines.append(str(max_evaluations))
        config_lines.append(str(init_type))
        config_lines.append(str(local_search_type))
        config_lines.append(str(path_relinking_mode))
        return config_lines

    def get_metrics_fields(self, result: Dict[str, Any], repetition):
        super().get_metrics_fields(result, repetition)

        numGenerations = str(
            result["numGenerations"]) if "numGenerations" in result else 'NaN'
        numEvaluations = str(
            result["numEvaluations"]) if "numEvaluations" in result else 'NaN'

        self.metrics_dictionary['NumGenerations'][repetition] = numGenerations
        self.metrics_dictionary['NumEvaluations'][repetition] = numEvaluations

