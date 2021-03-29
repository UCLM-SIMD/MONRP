from random import *
import uuid
import copy

class Individual:
    def __init__(self, genes, objectives):
        self.id = uuid.uuid4()
        self.score = None
        self.cost = None
        self.total_score = None
        self.genes = copy.deepcopy(genes)
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.objectives = copy.deepcopy(objectives)
        # if random is True:
        # print("ran")
        # self.initRandom()

    def evaluate_fitness(self):
        score = 0
        cost = 0
        for gen in self.genes:
            if gen.included == 1:
                score += gen.value
                cost += gen.estimation

        self.score = score
        self.cost = cost
        self.total_score = score / (cost + 1)

        self.objectives[0].value = self.score
        self.objectives[1].value = self.cost

    def initRandom(self):
        for r in self.genes:
            r.included = randint(0, 1)

    def dominates(self, other_individual):
        self.evaluate_fitness()
        other_individual.evaluate_fitness()
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            # minimize cost
            if first.value is None or second.value is None:
                print(self)
                print(first.value)
                print(second.value)
            if first.is_minimization():
                and_condition = and_condition and first.value <= second.value
                or_condition = or_condition or first.value < second.value
            else:
                # maximize score
                and_condition = and_condition and first.value >= second.value
                or_condition = or_condition or first.value > second.value
        return (and_condition and or_condition)

    def __str__(self):
        s = 'id: ' + str(self.id) + ',\nscore: ' + str(self.score) + ',\ncost: ' + str(
            self.cost) + ',\ntotal_score: ' + str(self.total_score) \
            + ',\ncrowding_distance: ' + str(self.crowding_distance) + ',\ndomination_count: '\
            + ',\nrank: ' + str(self.rank)+ str(self.domination_count) + '\n genes: [\n'
        for i in self.genes:
            s += str(i) + '\n'
        s += "]"
        return s

    def __eq__(self, other_ind):
        return(other_ind.objectives[0].value == self.objectives[0].value and \
					other_ind.objectives[1].value == self.objectives[1].value \
               and self.print_genes()==other_ind.print_genes())

    def __hash__(self):
        return hash(('genes', self.print_genes()))

    def print_genes(self):
        return_genes=""
        for i in range(0, len(self.genes)):
            return_genes+=str(self.genes[i].included)+","
        return return_genes
