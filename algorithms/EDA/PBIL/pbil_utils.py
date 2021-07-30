import random
from models.population import Population


class PBILUtils():
    def __init__(self, random_seed, population_length=20, selection_candidates=2, crossover_prob=0.9, mutation_prob=0.1):
        super().__init__(random_seed, population_length,
                         selection_candidates, crossover_prob, mutation_prob)

    # GENERATE DATASET PROBLEM------------------------------------------------------------------
    def generate_dataset_problem(self, dataset_name):
        return super().generate_dataset_problem(dataset_name)

    # GENERATE STARTING POPULATION------------------------------------------------------------------
    def generate_starting_population(self):
        return super().generate_starting_population()

    # EVALUATION------------------------------------------------------------------
    def evaluate(self, population, best_individual):
        super().evaluate(population, best_individual)

    # LAST GENERATION ENHANCE------------------------------------------------------------------
    def calculate_last_generation_with_enhance(self, best_generation, best_generation_avgValue, num_generation, population):
        return super().calculate_last_generation_with_enhance(best_generation, best_generation_avgValue, num_generation, population)


    # 




    # REPLACEMENT------------------------------------------------------------------
    def replacement_elitism(self, population, newpopulation):
        # print("POP----------------------------------------------------------")
        # encontrar mejor individuo de poblacion
        best_individual = None
        best_individual_total_score = 0
        for ind in population:
            # print(ind)
            if (ind.total_score > best_individual_total_score):
                best_individual_total_score = ind.total_score
                best_individual = ind
        # print("ind----------------------------------------------------------")
        # print(best_individual)
        # print("NEWPOP----------------------------------------------------------")

        # encontrar indice del peor individuo de nueva poblacion
        newpopulation_replaced = Population()
        # = copy.deepcopy(newpopulation)
        newpopulation_replaced.extend(newpopulation.population)

        worst_individual_total_score = float('inf')
        worst_individual_index = None
        for ind in newpopulation_replaced:
            # print(ind)
            if (ind.total_score < worst_individual_total_score):
                worst_individual_total_score = ind.total_score
                worst_individual_index = newpopulation_replaced.index(ind)

        # reemplazar peor individuo por el mejor de poblacion antigua
        # print("index------------ ", worst_individual_index)
        # print(newpopulation_replaced[worst_individual_index])
        newpopulation_replaced.set(worst_individual_index, best_individual)
        # print(newpopulation_replaced[worst_individual_index])
        # print("end")
        return newpopulation_replaced
