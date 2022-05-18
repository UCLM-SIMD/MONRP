from typing import Any, Dict, List
from algorithms.abstract_algorithm.abstract_executer import AbstractExecuter


class UMDAExecuter(AbstractExecuter):
    """Specific umda implementation of executer.
    """

    def __init__(self, algorithm, execs):
        """Init method extends config and metrics fields with specific umda algorithm data
        """
        from algorithms.EDA.UMDA.umda_algorithm import UMDAAlgorithm
<<<<<<< HEAD
        super().__init__(algorithm, num_execs=execs)
=======
        super().__init__(algorithm, excecs=execs)
>>>>>>> 19c7836f (ahora todos los resultados se almacenan en results.json con un id unico para cada conjunto de parametros de lanzamiento)
        self.algorithm: UMDAAlgorithm
        self.algorithm_type: str = "umda"

        self.config_fields.extend(["Population Length", "MaxGenerations", "MaxEvaluations",
                                   "Selected Individuals", "Selection Scheme", "Replacement Scheme"])

       # self.metrics_fields.extend(
        #    ["NumGenerations", "NumEvaluations", ])

        self.metrics_dictionary["NumGenerations"] = [None] * int(execs)
        self.metrics_dictionary["NumEvaluations"] = [None] * int(execs)

    def get_config_fields(self,) -> List[str]:
        """UMDA algorithm executer extends metrics fields read from the execution
        """
        config_lines: List[str] = super().get_config_fields()

        population_length = self.algorithm.population_length
        max_generations = self.algorithm.max_generations
        max_evaluations = self.algorithm.max_evaluations
        selected_individuals = self.algorithm.selected_individuals
        selection_scheme = self.algorithm.selection_scheme
        replacement_scheme = self.algorithm.replacement_scheme

        #config_lines.append(str(population_length))
        #config_lines.append(str(max_generations))
        #config_lines.append(str(max_evaluations))
        #config_lines.append(str(selected_individuals))
        #config_lines.append(str(selection_scheme))
        #config_lines.append(str(replacement_scheme))



        #return config_lines

    def get_metrics_fields(self, result: Dict[str, Any], repetition) -> List[str]:
        """UMDA algorithm executer extends metrics fields read from the execution
        """
        metrics_fields: List[str] = super().get_metrics_fields(result, repetition)

        numGenerations = result["numGenerations"] if "numGenerations" in result else 'NaN'
        numEvaluations = result["numEvaluations"] if "numEvaluations" in result else 'NaN'

        #metrics_fields.append(str(numGenerations))
        #metrics_fields.append(str(numEvaluations))
        self.metrics_dictionary['NumGenerations'][repetition] = numGenerations
        self.metrics_dictionary['NumEvaluations'][repetition] = numEvaluations

        return metrics_fields
