from typing import Any, Dict, List
from algorithms.abstract_algorithm.abstract_executer import AbstractExecuter


class FEDAExecuter(AbstractExecuter):
    """Specific umda implementation of executer.
    """

<<<<<<< HEAD
<<<<<<< HEAD
    def __init__(self, algorithm, execs:int):
        """Init method extends config and metrics fields with specific feda algorithm data
        """
        from algorithms.EDA.FEDA.feda_algorithm import FEDAAlgorithm
        super().__init__(algorithm,execs)
=======
    def __init__(self, algorithm):
        """Init method extends config and metrics fields with specific feda algorithm data
        """
        from algorithms.EDA.FEDA.feda_algorithm import FEDAAlgorithm
        super().__init__(algorithm)
>>>>>>> bd41d390 (first version of FEDA (Fixed-structure EDA) finished)
=======
    def __init__(self, algorithm, execs:int):
        """Init method extends config and metrics fields with specific feda algorithm data
        """
        from algorithms.EDA.FEDA.feda_algorithm import FEDAAlgorithm
        super().__init__(algorithm,execs)
>>>>>>> 032bf379 (executer_driver extendido para lanzar FEDA)
        self.algorithm: FEDAAlgorithm
        self.algorithm_type: str = "feda"

        self.config_fields.extend(["Population Length", "MaxGenerations", "MaxEvaluations",
                                   "Selected Individuals", "Selection Scheme", "Replacement Scheme"])

        self.metrics_fields.extend(
            ["NumGenerations", "NumEvaluations", ])

<<<<<<< HEAD
<<<<<<< HEAD
        self.metrics_dictionary["NumGenerations"] = [None] * int(execs)
        self.metrics_dictionary["NumEvaluations"] = [None] * int(execs)

=======
>>>>>>> bd41d390 (first version of FEDA (Fixed-structure EDA) finished)
=======
        self.metrics_dictionary["NumGenerations"] = [None] * int(execs)
        self.metrics_dictionary["NumEvaluations"] = [None] * int(execs)

>>>>>>> 032bf379 (executer_driver extendido para lanzar FEDA)
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

<<<<<<< HEAD
<<<<<<< HEAD
    def get_metrics_fields(self, result: Dict[str, Any], repetition) -> List[str]:
        """FEDA algorithm executer extends metrics fields read from the execution
        """
        metrics_fields: List[str] = super().get_metrics_fields(result, repetition=repetition)

        numGenerations = result["numGenerations"] if "numGenerations" in result else 'NaN'
        numEvaluations = result["numEvaluations"] if "numEvaluations" in result else 'NaN'
<<<<<<< HEAD

        #metrics_fields.append(str(numGenerations))
        #metrics_fields.append(str(numEvaluations))

        self.metrics_dictionary['NumGenerations'][repetition] = numGenerations
        self.metrics_dictionary['NumEvaluations'][repetition] = numEvaluations
=======
    def get_metrics_fields(self, result: Dict[str, Any]) -> List[str]:
=======
    def get_metrics_fields(self, result: Dict[str, Any], repetition) -> List[str]:
>>>>>>> 032bf379 (executer_driver extendido para lanzar FEDA)
        """FEDA algorithm executer extends metrics fields read from the execution
        """
        metrics_fields: List[str] = super().get_metrics_fields(result, repetition=repetition)

        numGenerations = str(
            result["numGenerations"]) if "numGenerations" in result else 'NaN'
        numEvaluations = str(
            result["numEvaluations"]) if "numEvaluations" in result else 'NaN'
=======
>>>>>>> 9617fc4f (extract_postMetrics.py computes and updates outputs .json with: gd+, unfr and reference pareto front.)

<<<<<<< HEAD
        metrics_fields.append(str(numGenerations))
        metrics_fields.append(str(numEvaluations))
>>>>>>> bd41d390 (first version of FEDA (Fixed-structure EDA) finished)
=======
        #metrics_fields.append(str(numGenerations))
        #metrics_fields.append(str(numEvaluations))

        self.metrics_dictionary['NumGenerations'][repetition] = numGenerations
        self.metrics_dictionary['NumEvaluations'][repetition] = numEvaluations
>>>>>>> 032bf379 (executer_driver extendido para lanzar FEDA)

        return metrics_fields
