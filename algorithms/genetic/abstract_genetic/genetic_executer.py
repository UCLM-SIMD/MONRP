from typing import Any, Dict, List
from algorithms.genetic.abstract_genetic.abstract_genetic_algorithm import AbstractGeneticAlgorithm

from algorithms.abstract_algorithm.abstract_executer import AbstractExecuter


class GeneticExecuter(AbstractExecuter):
    """Specific genetic implementation of executer.
    """

    def __init__(self, algorithm: AbstractGeneticAlgorithm, execs: int):
        """Init method extends config and metrics fields with specific genetic algorithm data
        """
        super().__init__(algorithm, execs)
        self.algorithm: AbstractGeneticAlgorithm = algorithm
        self.algorithm_type: str = "genetic"

        self.config_fields.extend(["Population Length", "MaxGenerations", "MaxEvaluations",
                                   "Selection Scheme", "Selection Candidates", "Crossover Scheme", "Crossover Probability",
                                   "Mutation Scheme", "Mutation Probability", "Replacement Scheme", ])

        self.metrics_fields.extend(
            ["NumGenerations", "NumEvaluations", "BestGeneration", ])
        self.metrics_dictionary["NumGenerations"] = [None] * int(execs)
        self.metrics_dictionary["NumEvaluations"] = [None] * int(execs)
        self.metrics_dictionary["BestGeneration"] = [None] * int(execs)

    def get_config_fields(self,) -> List[str]:
        """Genetic algorithm executer extends config fields read from the execution
        """
        config_lines: List[str] = super().get_config_fields()

        population_length = self.algorithm.population_length
        generations = self.algorithm.max_generations
        max_evaluations = self.algorithm.max_evaluations
        selection = self.algorithm.selection_scheme
        selection_candidates = self.algorithm.selection_candidates
        crossover = self.algorithm.crossover_scheme
        crossover_prob = self.algorithm.crossover_prob
        mutation = self.algorithm.mutation_scheme
        mutation_prob = self.algorithm.mutation_prob
        replacement = self.algorithm.replacement_scheme

        config_lines.append(str(population_length))
        config_lines.append(str(generations))
        config_lines.append(str(max_evaluations))
        config_lines.append(str(selection))
        config_lines.append(str(selection_candidates))
        config_lines.append(str(crossover))
        config_lines.append(str(mutation))
        config_lines.append(str(crossover_prob))
        config_lines.append(str(mutation))
        config_lines.append(str(mutation_prob))
        config_lines.append(str(replacement))
        return config_lines

    def get_metrics_fields(self, result: Dict[str, Any], repetition) -> List[str]:
        """Genetic algorithm executer extends metrics fields read from the execution
        """
        metrics_fields: List[str] = super().get_metrics_fields(result, repetition)

        numGenerations = result["numGenerations"] if "numGenerations" in result else 'NaN'
        numEvaluations = result["numEvaluations"] if "numEvaluations" in result else 'NaN'
        bestGeneration = result["bestGeneration"] if "bestGeneration" in result else 'NaN'

       # metrics_fields.append(str(numGenerations))
       # metrics_fields.append(str(numEvaluations))
       # metrics_fields.append(str(bestGeneration))

        self.metrics_dictionary['NumGenerations'][repetition] = numGenerations
        self.metrics_dictionary['NumEvaluations'][repetition] = numEvaluations
        self.metrics_dictionary['BestGeneration'][repetition] = bestGeneration

        return metrics_fields
