from typing import Any


class Hyperparameter:
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value


HYPERPARAMETER_TRANSLATIONS = {
    "population_length": "Population Length",
    "max_generations": "MaxGenerations",
    "max_evaluations": "MaxEvaluations",
    "selection_scheme": "Selection Scheme",
    "selection_candidates": "Selection Candidates",
    "crossover_scheme": "Crossover Scheme",
    "crossover_prob": "Crossover Probability",
    "mutation_scheme": "Mutation Scheme",
    "mutation_prob": "Mutation Probability",
    "replacement_scheme": "Replacement Scheme",
    "solutions_per_iteration": "Population Length",
    "iterations": "MaxGenerations",
    "init_type": "Initialization Type",
    "local_search_type": "Local Search Type",
    "path_relinking_mode": "Path Relinking",
    "selected_individuals": "Selected Individuals",
    "learning_rate": "Learning Rate",
    "mutation_shift": "Mutation Shift",
}


def generate_hyperparameter(code: str, value: Any) -> Hyperparameter:
    return Hyperparameter(HYPERPARAMETER_TRANSLATIONS[code], value)
