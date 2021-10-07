import numpy as np
from models.solution import Solution
from models.problem import Problem
from datasets.dataset_gen_generator import generate_dataset_genes


def format_population(population,dataset):
    genes, _ = generate_dataset_genes(dataset.id)
    problem = Problem(genes, ["MAX", "MIN"])
    final_nds_formatted = []

    for solution in population:
        # print(solution)
        individual = Solution(problem.genes, problem.objectives)
        for b in np.arange(len(individual.genes)):
            individual.genes[b].included = solution.selected[b]
        individual.evaluate_fitness()
        final_nds_formatted.append(individual)
    population = final_nds_formatted
    return population
