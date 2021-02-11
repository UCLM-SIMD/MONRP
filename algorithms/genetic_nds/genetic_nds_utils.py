import random
from models.population import Population
#from scipy.spatial import distance
import math

class GeneticNDSUtils:
    def __init__(self, problem, random_seed,selection_candidates=2, crossover_prob=0.9,mutation_prob=0.1):
        self.problem=problem
        self.random_seed=random_seed
        random.seed(self.random_seed)
        self.selection_candidates=selection_candidates
        self.crossover_prob=crossover_prob
        self.mutation_prob=mutation_prob

    #SELECTION------------------------------------------------------------------
    def selection_tournament(self,population):
      #self.num_candidates=2
      new_population=Population()
      #crear tantos individuos como tamaño de la poblacion
      for i in range(0,len(population)):
        best_candidate=None
        best_total_score=0
        #elegir individuo entre X num de candidatos aleatorios
        for j in range(0,self.selection_candidates):
          random_index=random.randint(0,len(population)-1)
          candidate=population.get(random_index)
          candidate.evaluate_fitness()
          #print(candidate)
          score=candidate.score
          cost=candidate.cost
          total_score=candidate.total_score
          #guardar el candidato con mejor score
          if( total_score > best_total_score):
            best_total_score=total_score
            best_candidate=candidate

        #insertar el mejor candidato del torneo en la nueva poblacion
        #print(best_candidate)
        new_population.append(best_candidate)

      #retornar nueva poblacion
      return new_population


    # CROSSOVER------------------------------------------------------------------
    def crossover_one_point(self, population):
        new_population = Population()

        # for i in range(0, len(population)-1):
        i = 0
        while i < len(population):
            # if last element is alone-> add it
            if i == len(population) - 1:
                new_population.append(population.get(i))

            else:
                # pair 2 parents -> crossover or add them and jump 1 index extra
                prob = random.random()
                if prob < self.crossover_prob:
                    offsprings = self.crossover_aux_one_point(population.get(i), population.get(i + 1))
                    # print(offsprings)
                    new_population.extend(offsprings)
                else:
                    new_population.extend([population.get(i), population.get(i + 1)])
                i += 1

            i += 1

        return new_population

    def crossover_aux_one_point(self,parent1, parent2):
        # print("-----------PARENTS:")
        # print(parent1)
        # print(parent2)
        chromosome_length = len(parent1.genes)
        # index aleatorio del punto de division para el cruce
        crossover_point = random.randint(1, chromosome_length - 1)
        # print('-------------crossover_point :', crossover_point )
        offspring_genes1 = parent1.genes[0:crossover_point] + parent2.genes[crossover_point:]
        # print("first slice")
        # for o in offspring_genes1:
        # print("gen: ",o)

        offspring_genes2 = parent2.genes[0:crossover_point] + parent1.genes[crossover_point:]

        # print("second slice")
        # for o in offspring_genes2:
        # print("gen: ",o)

        offspring1 = self.problem.generate_individual(offspring_genes1)
        offspring2 = self.problem.generate_individual(offspring_genes2)

        # print("-----------OFFSPRINGS:")
        # print(offspring1)
        # print(offspring2)

        return offspring1, offspring2


    # MUTATION------------------------------------------------------------------
    def mutation(self,population):
        new_population = Population()
        new_population.extend(population.population)

        for individual in new_population:
            prob = random.random()
        if prob < self.mutation_prob:
            # print(individual)
            chromosome_length = len(individual.genes)
            # print(prob)
            mutation_point = random.randint(0, chromosome_length - 1)
            if individual.genes[mutation_point].included == 0:
                individual.genes[mutation_point].included = 1
            else:
                individual.genes[mutation_point].included = 0

        return new_population

    # REPLACEMENT------------------------------------------------------------------
    def replacement_elitism(self,population, newpopulation):
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

        # HYPERVOLUME------------------------------------------------------------------
    def calculate_hypervolume(self, population):
        # obtener minimos y maximos de cada objetivo
        objectives_diff = []
        for i in range(0, len(population.get(0).objectives)):
            aux_min = float('inf')
            aux_max = 0
            for ind in population:  ##############################################
                if ind.objectives[i].value < aux_min:
                    aux_min = ind.objectives[i].value
                if ind.objectives[i].value > aux_max:
                    aux_max = ind.objectives[i].value

            objectives_diff.append(aux_max - aux_min)

        # calcular hypervolume
        hypervolume = 1
        for i in range(0, len(objectives_diff)):
            hypervolume *= objectives_diff[i]

        return hypervolume

    # SPREAD------------------------------------------------------------------
    def eudis2(self, v1, v2):
        return math.dist(v1, v2)
        #return distance.euclidean(v1, v2)

    def calculate_spread(self, population):
        MIN_OBJ1 = 0
        MIN_OBJ2 = 0
        MAX_OBJ1 = 25  # max_importancia_Stakeholder * max_prioridad_pbi_para_Stakeholder
        MAX_OBJ2 = 40  # max estimacion de pbi
        df = None
        dl = None
        davg = None
        sum_dist = None
        N = len(population)
        spread = None

        first_solution = population.get(0)
        last_solution = population.get(len(population) - 1)

        # obtener first_extreme=[score=0 (worst),cost=0 (best)] y last_extreme=[score=MAX_SCORE (best),cost=MAX_COST (worst)]
        first_extreme = [MIN_OBJ1, MIN_OBJ2]
        last_extreme = [MAX_OBJ1, MAX_OBJ2]

        df = self.eudis2([first_solution.objectives[0].value, first_solution.objectives[1].value], first_extreme)
        dl = self.eudis2([last_solution.objectives[0].value, last_solution.objectives[1].value], last_extreme)

        # calcular media de todas las distancias entre puntos
        davg = 0
        dist_count = 0
        for i in range(0, len(population)):
            for j in range(0, len(population)):
                # no calcular distancia de un punto a si mismo
                if i != j:
                    dist_count += 1
                    davg += self.eudis2(
                        [population.get(i).objectives[0].value, population.get(i).objectives[1].value],
                        [population.get(j).objectives[0].value, population.get(j).objectives[1].value])
        # media=distancia total / numero de distancias
        davg /= dist_count

        # calcular sumatorio(i=1->N-1) |di-davg|
        sum_dist = 0
        for i in range(0, len(population) - 1):
            di = self.eudis2([population.get(i).objectives[0].value, population.get(i).objectives[1].value],
                             [population.get(i + 1).objectives[0].value, population.get(i + 1).objectives[1].value])
            sum_dist += abs(di - davg)

        # formula spread
        spread = (df + dl + sum_dist) / (df + dl + (N - 1) * davg)
        return spread