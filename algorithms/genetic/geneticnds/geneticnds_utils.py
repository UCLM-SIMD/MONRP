from algorithms.genetic.abstract_genetic.basegenetic_utils import BaseGeneticUtils
import random
from models.population import Population


class GeneticNDSUtils(BaseGeneticUtils):
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

    # SELECTION------------------------------------------------------------------
    def selection_tournament(self, population):
        new_population = Population()
        # crear tantos individuos como tamaño de la poblacion
        for i in range(0, len(population)):
            best_candidate = None
            best_total_score = 0
            # elegir individuo entre X num de candidatos aleatorios
            for j in range(0, self.selection_candidates):
                random_index = random.randint(0, len(population)-1)
                candidate = population.get(random_index)
                candidate.evaluate_fitness()
                score = candidate.score
                cost = candidate.cost
                total_score = candidate.total_score
                # guardar el candidato con mejor score
                if(total_score > best_total_score):
                    best_total_score = total_score
                    best_candidate = candidate

            # insertar el mejor candidato del torneo en la nueva poblacion
            new_population.append(best_candidate)

        # retornar nueva poblacion
        return new_population

    # CROSSOVER------------------------------------------------------------------

    def crossover_one_point(self, population):
        new_population = Population()
        i = 0
        while i < len(population):
            # if last element is alone-> add it
            if i == len(population) - 1:
                new_population.append(population.get(i))

            else:
                # pair 2 parents -> crossover or add them and jump 1 index extra
                prob = random.random()
                if prob < self.crossover_prob:
                    offsprings = self.crossover_aux_one_point(
                        population.get(i), population.get(i + 1))
                    # print(offsprings)
                    new_population.extend(offsprings)
                else:
                    new_population.extend(
                        [population.get(i), population.get(i + 1)])
                i += 1

            i += 1

        return new_population

    def crossover_aux_one_point(self, parent1, parent2):
        chromosome_length = len(parent1.genes)
        # index aleatorio del punto de division para el cruce
        crossover_point = random.randint(1, chromosome_length - 1)
        offspring_genes1 = parent1.genes[0:crossover_point] + \
            parent2.genes[crossover_point:]

        offspring_genes2 = parent2.genes[0:crossover_point] + \
            parent1.genes[crossover_point:]

        offspring1 = self.problem.generate_individual(offspring_genes1)
        offspring2 = self.problem.generate_individual(offspring_genes2)

        return offspring1, offspring2

    # MUTATION------------------------------------------------------------------

    def mutation_flip1bit(self, population):
        new_population = Population()
        new_population.extend(population.population)

        for individual in new_population:
            prob = random.random()
            if prob < self.mutation_prob:
                chromosome_length = len(individual.genes)
                mutation_point = random.randint(0, chromosome_length - 1)
                if individual.genes[mutation_point].included == 0:
                    individual.genes[mutation_point].included = 1
                else:
                    individual.genes[mutation_point].included = 0

        return new_population

    def mutation_flipeachbit(self, population):
        new_population = Population()
        new_population.extend(population.population)

        for individual in new_population:
            for gen in individual.genes:
                prob = random.random()
                if prob < self.mutation_prob:
                    if gen.included == 0:
                        gen.included = 1
                    else:
                        gen.included = 0

        return new_population

    # REPLACEMENT------------------------------------------------------------------
    def replacement_elitism(self, population, newpopulation):
        # encontrar mejor individuo de poblacion
        best_individual = None
        best_individual_total_score = 0
        for ind in population:
            if (ind.total_score > best_individual_total_score):
                best_individual_total_score = ind.total_score
                best_individual = ind

        # encontrar indice del peor individuo de nueva poblacion
        newpopulation_replaced = Population()
        newpopulation_replaced.extend(newpopulation.population)

        worst_individual_total_score = float('inf')
        worst_individual_index = None
        for ind in newpopulation_replaced:
            if (ind.total_score < worst_individual_total_score):
                worst_individual_total_score = ind.total_score
                worst_individual_index = newpopulation_replaced.index(ind)

        # reemplazar peor individuo por el mejor de poblacion antigua
        newpopulation_replaced.set(worst_individual_index, best_individual)
        return newpopulation_replaced
