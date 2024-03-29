from typing import Any, Dict, List
from algorithms.abstract_algorithm.abstract_executer import AbstractExecuter


class FEDAExecuter(AbstractExecuter):
    """Specific umda implementation of executer.
    """

    def __init__(self, algorithm, execs:int):
        """Init method extends config and metrics fields with specific feda algorithm data
        """
        from algorithms.EDA.FEDA.feda_algorithm import FEDAAlgorithm
        super().__init__(algorithm,execs)
        self.algorithm: FEDAAlgorithm
        self.algorithm_type: str = "feda"

        self.config_fields.extend(["Population Length", "MaxGenerations", "MaxEvaluations",
                                   "Selected Individuals", "Selection Scheme", "Replacement Scheme"])

        self.metrics_fields.extend(
            ["NumGenerations", "NumEvaluations", ])

        self.metrics_dictionary["NumGenerations"] = [None] * int(execs)
        self.metrics_dictionary["NumEvaluations"] = [None] * int(execs)

    def get_config_fields(self,) -> List[str]:
        """FEDA algorithm executer extends metrics fields read from the execution
        """
        config_lines: List[str] = super().get_config_fields()

        population_length = self.algorithm.population_length
        max_generations = self.algorithm.max_generations
        max_evaluations = self.algorithm.max_evaluations
        selected_individuals = self.algorithm.selected_individuals
        selection_scheme = self.algorithm.selection_scheme
        replacement_scheme = self.algorithm.replacement_scheme

        config_lines.append(str(population_length))
        config_lines.append(str(max_generations))
        config_lines.append(str(max_evaluations))
        config_lines.append(str(selected_individuals))
        config_lines.append(str(selection_scheme))
        config_lines.append(str(replacement_scheme))
        return config_lines

    def get_metrics_fields(self, result: Dict[str, Any], repetition) -> List[str]:
        """FEDA algorithm executer extends metrics fields read from the execution
        """
        metrics_fields: List[str] = super().get_metrics_fields(result, repetition=repetition)

        numGenerations = result["numGenerations"] if "numGenerations" in result else 'NaN'
        numEvaluations = result["numEvaluations"] if "numEvaluations" in result else 'NaN'

        #metrics_fields.append(str(numGenerations))
        #metrics_fields.append(str(numEvaluations))

        self.metrics_dictionary['NumGenerations'][repetition] = numGenerations
        self.metrics_dictionary['NumEvaluations'][repetition] = numEvaluations

        return metrics_fields
