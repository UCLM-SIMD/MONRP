from typing import Any, Dict, List
from algorithms.abstract_algorithm.abstract_executer import AbstractExecuter


class UMDAExecuter(AbstractExecuter):
    """Specific umda implementation of executer.
    """

    def __init__(self, algorithm, execs):
        """Init method extends config and metrics fields with specific umda algorithm data
        """
        from algorithms.EDA.UMDA.umda_algorithm import UMDAAlgorithm
        super().__init__(algorithm, num_execs=execs)
        self.algorithm: UMDAAlgorithm
        self.algorithm_type: str = "umda"

        self.config_fields.extend(["Population Length", "MaxGenerations", "MaxEvaluations",
                                   "Selected Individuals", "Selection Scheme", "Replacement Scheme"])

        self.metrics_dictionary["NumGenerations"] = [None] * int(execs)
        self.metrics_dictionary["NumEvaluations"] = [None] * int(execs)

    def get_metrics_fields(self, result: Dict[str, Any], repetition) -> List[str]:
        """UMDA algorithm executer extends metrics fields read from the execution
        """
        metrics_fields: List[str] = super(
        ).get_metrics_fields(result, repetition)

        numGenerations = result["numGenerations"] if "numGenerations" in result else 'NaN'
        numEvaluations = result["numEvaluations"] if "numEvaluations" in result else 'NaN'

        self.metrics_dictionary['NumGenerations'][repetition] = numGenerations
        self.metrics_dictionary['NumEvaluations'][repetition] = numEvaluations

        return metrics_fields
