import random
from evaluation.format_population import format_population
from algorithms.abstract_default.evaluation_exception import EvaluationLimit
from algorithms.genetic.abstract_genetic.basegenetic_algorithm import BaseGeneticAlgorithm
from algorithms.genetic.nsgaii.nsgaii_executer import NSGAIIExecuter
from algorithms.genetic.nsgaii.nsgaii_utils import NSGAIIUtils
from models.population import Population
import copy
import time


# TODO NSGAIIALGORITHM -> NSGAII y reescribir ficheros output
class NSGAIIAlgorithm(BaseGeneticAlgorithm):
    def __init__(self, dataset_name="test", random_seed=None, population_length=20, max_generations=1000, max_evaluations=0,
                 selection="tournament", selection_candidates=2,
                 crossover="onepoint", crossover_prob=0.9,
                 mutation="flipeachbit", mutation_prob=0.1,
                 replacement="elitism", debug_mode=False, tackle_dependencies=False):

        super().__init__(dataset_name, random_seed, debug_mode, tackle_dependencies,
                         population_length, max_generations, max_evaluations)

        # self.utils = NSGAIIUtils(
        #    random_seed, population_length, selection_candidates, crossover_prob, mutation_prob)
        self.executer = NSGAIIExecuter(algorithm=self)
        # self.problem, self.dataset = self.utils.generate_dataset_problem(
        #    dataset_name=dataset_name)
        self.dataset_name = dataset_name

        #self.random_seed = random_seed
        #self.population_length = population_length
        #self.max_generations = max_generations
        #self.max_evaluations = max_evaluations

        self.selection_scheme = selection
        self.selection_candidates = selection_candidates
        self.crossover_scheme = crossover
        self.crossover_prob = crossover_prob
        self.mutation_scheme = mutation
        self.mutation_prob = mutation_prob
        self.replacement_scheme = replacement

        self.population = None
        self.best_generation_avgValue = None
        self.best_generation = None

        self.num_evaluations: int = 0
        self.num_generations: int = 0
        self.best_individual = None

        #self.debug_mode = debug_mode
        #self.tackle_dependencies = tackle_dependencies

        #self.fast_nondominated_sort = self.utils.fast_nondominated_sort
        #self.calculate_crowding_distance = self.utils.calculate_crowding_distance
        #self.crowding_operator = self.utils.crowding_operator
        #self.evaluate = self.utils.evaluate
        #self.calculate_last_generation_with_enhance = self.utils.calculate_last_generation_with_enhance
        #self.generate_starting_population = self.utils.generate_starting_population

        #self.repair_population_dependencies = self.utils.repair_population_dependencies

        if selection == "tournament":
            self.selection = self.selection_tournament

        if crossover == "onepoint":
            self.crossover = self.crossover_one_point

        if mutation == "flip1bit":
            self.mutation = self.mutation_flip1bit
        elif mutation == "flipeachbit":
            self.mutation = self.mutation_flipeachbit

        if replacement == "elitism":
            self.replacement = self.replacement_elitism
        else:
            self.replacement = self.replacement_elitism

        self.file: str = str(self.__class__.__name__)+"-"+str(dataset_name)+"-"+str(random_seed)+"-"+str(population_length)+"-" +\
            str(max_generations) + "-"+selection+"-"+str(selection_candidates)+"-" +\
            str(crossover)+"-"+str(crossover_prob)+"-"+str(mutation) + \
            "-"+str(mutation_prob)+"-"+str(replacement)+".txt"
        # + "-"+str(max_evaluations) TODO

    def get_name(self):
        return "NSGA-II+"+str(self.population_length)+"+"+str(self.max_generations)+"+"+str(self.max_evaluations)\
            + "+"+str(self.crossover_prob)\
            + "+"+str(self.mutation_scheme)+"+"+str(self.mutation_prob)

    # def evaluate(self, population, best_individual):
    #    #super().evaluate(population, best_individual)
    #    try:
    #        best_score = 0
    #        new_best_individual = None
    #        for ind in population:
    #            ind.evaluate_fitness()
    #            self.add_evaluation(population)#############
    #            if ind.total_score > best_score:
    #                new_best_individual = copy.deepcopy(ind)
    #                best_score = ind.total_score
    #        if best_individual is not None:
    #            if new_best_individual.total_score > best_individual.total_score:
    #                best_individual = copy.deepcopy(new_best_individual)
    #        else:
    #            best_individual = copy.deepcopy(new_best_individual)
    #    except EvaluationLimit:
    #        pass

    def add_evaluation(self, new_population):
        self.num_evaluations += 1
        # if(self.num_evaluations >= self.max_evaluations):
        if (self.stop_criterion(self.num_generations, self.num_evaluations)):
            # acciones:
            self.returned_population = copy.deepcopy(new_population)
            self.fast_nondominated_sort(self.returned_population)
            self.best_generation, self.best_generation_avgValue = self.calculate_last_generation_with_enhance(
                self.best_generation, self.best_generation_avgValue, self.num_generations, self.returned_population)
            raise EvaluationLimit

    def reset(self):
        super().reset()
        self.returned_population = None

    # RUN ALGORITHM------------------------------------------------------------------
    def run(self):
        self.reset()
        paretos = []
        start = time.time()

        # inicializacion del nsgaii
        self.population = self.generate_starting_population()
        self.returned_population = copy.deepcopy(self.population)
        self.evaluate(self.population, self.best_individual)
        # self.num_evaluations+=len(self.population)

        # ordenar por NDS y crowding distance
        self.population, fronts = self.fast_nondominated_sort(self.population)
        for front in fronts:
            self.calculate_crowding_distance(front)

        # crear hijos
        offsprings = self.selection(self.population)
        offsprings = self.crossover(offsprings)
        offsprings = self.mutation(offsprings)
        # offsprings = self.replacement(self.population, offsprings)

        # or not(num_generations > (self.best_generation+20)):
        # while (num_generations < self.max_generations):
        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                self.population.extend(offsprings)
                self.evaluate(self.population, self.best_individual)
                # self.num_evaluations+=len(self.population)

                self.population, fronts = self.fast_nondominated_sort(self.population)
                new_population = []
                front_num = 0

                # till parent population is filled, calculate crowding distance in Fi, include i-th non-dominated front in parent pop
                while len(new_population) + len(fronts[front_num]) <= self.population_length:
                    self.calculate_crowding_distance(
                        fronts[front_num])
                    new_population.extend(fronts[front_num])
                    front_num += 1

                # ordenar los individuos del ultimo front por crowding distance y agregar los X que falten para completar la poblacion
                self.calculate_crowding_distance(
                    fronts[front_num])

                # sort in descending order using >=n
                fronts[front_num].sort(
                    key=lambda individual: individual.crowding_distance, reverse=True)

                # choose first N elements of Pt+1
                new_population.extend(
                    fronts[front_num][0:self.population_length - len(new_population)])
                self.population = copy.deepcopy(new_population)
                # ordenar por NDS y crowding distance
                self.population, fronts = self.fast_nondominated_sort(self.population)
                for front in fronts:
                    self.calculate_crowding_distance(front)

                # use selection,crossover and mutation to create a new population Qt+1
                offsprings = self.selection(self.population)
                offsprings = self.crossover(offsprings)
                offsprings = self.mutation(offsprings)
                # offsprings = self.replacement(self.population, offsprings)

                # repair population if dependencies tackled:
                if(self.tackle_dependencies):
                    # offsprings = self.repair_population_dependencies(
                    #    offsprings)
                    self.population = self.repair_population_dependencies(
                        self.population)
                    fronts[0] = self.repair_population_dependencies(
                        fronts[0])

                self.returned_population = copy.deepcopy(self.population)
                self.best_generation, self.best_generation_avgValue = self.calculate_last_generation_with_enhance(
                    self.best_generation, self.best_generation_avgValue, self.num_generations, self.returned_population)

                self.num_generations += 1

                if self.debug_mode:
                    paretos.append(self.nds)

                # mostrar por pantalla
                # if num_generations % 100 == 0:
                #	print("Nº Generations: ", num_generations)

        except EvaluationLimit:
            pass

        end = time.time()

        return {"population": fronts[0],
                "time": end - start,
                "best_individual": self.best_individual,
                "bestGeneration": self.best_generation,
                "numGenerations": self.num_generations,
                "numEvaluations": self.num_evaluations,
                "paretos": paretos
                }

    # SELECTION------------------------------------------------------------------
    def selection_tournament(self, population):
        #self.num_candidates = 2
        new_population = []
        # crear tantos individuos como tamaño de la poblacion
        for i in range(0, len(population)):
            best_candidate = None
            # elegir individuo entre X num de candidatos aleatorios
            for j in range(0, self.selection_candidates):
                random_index = random.randint(0, len(population) - 1)
                candidate = population[random_index]

                # guardar el candidato que mejor crowding tiene
                if (best_candidate is None or self.crowding_operator(candidate, best_candidate) == 1):
                    best_candidate = copy.deepcopy(candidate)

            # insertar el mejor candidato del torneo en la nueva poblacion
            new_population.append(best_candidate)

        # retornar nueva poblacion
        return new_population

    # CROSSOVER------------------------------------------------------------------
    # def crossover_one_point(self, population):
    #    new_population = []
#
    #    # for i in range(0, len(population)-1):
    #    i = 0
    #    while i < len(population):
    #        # if last element is alone-> add it
    #        if i == len(population) - 1:
    #            new_population.append(population[i])
    #        else:
    #            # pair 2 parents -> crossover or add them and jump 1 index extra
    #            prob = random.random()
    #            if prob < self.crossover_prob:
    #                offsprings = self.crossover_aux_one_point(
    #                    population[i], population[i+1])
    #                new_population.extend(offsprings)
    #            else:
    #                new_population.extend(
    #                    [population[i], population[i+1]])
    #            i += 1
#
    #        i += 1
#
    #    return new_population

#    def crossover_aux_one_point(self, parent1, parent2):
#        chromosome_length = len(parent1.genes)
#
#        # index aleatorio del punto de division para el cruce
#        crossover_point = random.randint(1, chromosome_length - 1)
#        offspring_genes1 = parent1.genes[0:crossover_point] + \
#            parent2.genes[crossover_point:]
#        offspring_genes2 = parent2.genes[0:crossover_point] + \
#            parent1.genes[crossover_point:]
#
#        offspring1 = self.problem.generate_individual(offspring_genes1,self.dataset.dependencies)
#        offspring2 = self.problem.generate_individual(offspring_genes2,self.dataset.dependencies)
#
#        return offspring1, offspring2

    # MUTATION------------------------------------------------------------------
#    def mutation_flip1bit(self, population):
#        new_population = Population()
#        new_population.extend(population.population)
#
#        for individual in new_population:
#            prob = random.random()
#            if prob < self.mutation_prob:
#                chromosome_length = len(individual.genes)
#                mutation_point = random.randint(0, chromosome_length - 1)
#                if individual.genes[mutation_point].included == 0:
#                    individual.genes[mutation_point].included = 1
#                else:
#                    individual.genes[mutation_point].included = 0
#
#        return new_population
#
#    def mutation_flipeachbit(self, population):
#        new_population = Population()
#        new_population.extend(population.population)
#
#        for individual in new_population:
#            for gen in individual.genes:
#                prob = random.random()
#                if prob < self.mutation_prob:
#                    if gen.included == 0:
#                        gen.included = 1
#                    else:
#                        gen.included = 0
#
#        return new_population

    # REPLACEMENT------------------------------------------------------------------
    def replacement_elitism(self, population, newpopulation):
        # encontrar mejor individuo de poblacion
        best_individual = None
        for ind in population:
            if (best_individual is None or self.crowding_operator(ind, best_individual) == 1):
                best_individual = copy.deepcopy(ind)

        newpopulation_replaced = []
        newpopulation_replaced.extend(newpopulation)
        worst_individual_index = None
        worst_individual = None
        for ind in newpopulation_replaced:
            if (worst_individual is None or self.crowding_operator(ind, worst_individual) == -1):
                worst_individual = copy.deepcopy(ind)
                worst_individual_index = newpopulation_replaced.index(ind)

        # reemplazar peor individuo por el mejor de poblacion antigua
        newpopulation_replaced[worst_individual_index] = best_individual
        return newpopulation_replaced

    # FAST NONDOMINATED SORT------------------------------------------------------------------
    def fast_nondominated_sort(self, population):
        fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                #if not individual.__eq__(other_individual):##########################################
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                fronts[0].append(individual)
        i = 0
        while len(fronts[i]) > 0:
            temp = []
            for individual in fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            fronts.append(temp)
        return population, fronts

    # CALCULATE CROWDING DISTANCE------------------------------------------------------------------
    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            front.sort(
                    key=lambda individual: individual.total_cost)
            # front[0].crowding_distance = 10 ** 9 #########################
            front[0].crowding_distance = float('inf')
            # front[solutions_num - 1].crowding_distance = 10 ** 9 #########################
            front[solutions_num - 1].crowding_distance = float('inf')
            m_values = [
                individual.total_cost for individual in front]
            scale = max(m_values) - min(
                m_values)  # aqui calcula la escala o diferencia entre el valor mayor y el menor, y la usa para dividir en el crowding distance
            if scale == 0:
                scale = 1
            for i in range(1, solutions_num - 1):
                front[i].crowding_distance += (
                    front[i + 1].total_cost - front[i - 1].total_cost) / scale

            front.sort(
                    key=lambda individual: individual.total_satisfaction)
            # front[0].crowding_distance = 10 ** 9 #########################
            front[0].crowding_distance = float('inf')
            # front[solutions_num - 1].crowding_distance = 10 ** 9 #########################
            front[solutions_num - 1].crowding_distance = float('inf')
            m_values = [
                individual.total_satisfaction for individual in front]
            scale = max(m_values) - min(
                m_values)  # aqui calcula la escala o diferencia entre el valor mayor y el menor, y la usa para dividir en el crowding distance
            if scale == 0:
                scale = 1
            for i in range(1, solutions_num - 1):
                front[i].crowding_distance += (
                    front[i + 1].total_satisfaction - front[i - 1].total_satisfaction) / scale



            #for m in range(len(front[0].objectives)):
            #    front.sort(
            #        key=lambda individual: individual.objectives[m].value)
            #    # front[0].crowding_distance = 10 ** 9 #########################
            #    front[0].crowding_distance = float('inf')
            #    # front[solutions_num - 1].crowding_distance = 10 ** 9 #########################
            #    front[solutions_num - 1].crowding_distance = float('inf')
            #    m_values = [
            #        individual.objectives[m].value for individual in front]
            #    scale = max(m_values) - min(
            #        m_values)  # aqui calcula la escala o diferencia entre el valor mayor y el menor, y la usa para dividir en el crowding distance
            #    if scale == 0:
            #        scale = 1
            #    for i in range(1, solutions_num - 1):
            #        front[i].crowding_distance += (
            #            front[i + 1].objectives[m].value - front[i - 1].objectives[m].value) / scale

    # CROWDING OPERATOR------------------------------------------------------------------
    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
            ((individual.rank == other_individual.rank) and (
                individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1
