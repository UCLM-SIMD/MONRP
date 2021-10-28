import random
import uuid
import copy


class Solution:
    def __init__(self, genes, objectives, dependencies):
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
        self.dependencies = dependencies

        self.max_cost = 0
        self.max_score = 0
        self.min_cost = 0
        self.min_score = 0
        for gen in self.genes:
            self.max_score += gen.value
            self.max_cost += gen.estimation
        self.max_totalscore = self.max_score/(self.min_cost+1)
        self.min_totalscore = self.min_score/(self.max_cost+1)

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
        self.total_score = score / (cost+1)
        # normalizar valores
        if (self.max_totalscore-self.min_totalscore) <= 0:
            self.total_score = 0
        else:
            self.total_score = (self.total_score-self.min_totalscore) / \
                (self.max_totalscore-self.min_totalscore)

        self.objectives[0].value = self.score
        self.objectives[1].value = self.cost

    def initRandom(self):
        for r in self.genes:
            r.included = random.randint(0, 1)

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

    def correct_dependencies(self):
        # for each included gene
        for gene_index in range(len(self.genes)):
            if(self.genes[gene_index].included == 1):
                # if has dependencies -> include all genes
                if self.dependencies[gene_index] is None:
                    continue
                for other_gene in self.dependencies[gene_index]:
                    self.genes[other_gene-1].included = 1


        #genes_len = len(self.genes)
        #g = Graph(genes_len)
        #for origin_index in range(len(self.dependencies)):
        #    if self.dependencies[origin_index] is not None:
        #        # print(origin_index)
        #        origin_dependencies = self.dependencies[origin_index]
        #        for dir_index in range(len(origin_dependencies)):
        #            direction = origin_dependencies[dir_index]
        #            # print(direction)
        #            g.addEdge(origin_index+1, direction)
#
        #genes_index_order = g.topologicalSort()
        #print(g.graph)
        #print(genes_index_order)
        #for gen_index in genes_index_order:
        #    if self.genes[gen_index-1].included and \
        #            self.dependencies[gen_index-1] is not None:
        #        for dep in self.dependencies[gen_index-1]:
        #            self.genes[dep-1].included = 1

    def __str__(self):
        s = 'id: ' + str(self.id) + ',\nscore: ' + str(self.score) + ',\ncost: ' + str(
            self.cost) + ',\ntotal_score: ' + str(self.total_score) \
            + ',\ncrowding_distance: ' + str(self.crowding_distance) + ',\ndomination_count: '\
            + ',\nrank: ' + str(self.rank) + \
            str(self.domination_count) + '\n genes: [\n'
        for i in self.genes:
            s += str(i) + '\n'
        s += "]"
        return s

    def __eq__(self, other_ind):
        return(other_ind.objectives[0].value == self.objectives[0].value and
               other_ind.objectives[1].value == self.objectives[1].value
               and self.print_genes() == other_ind.print_genes())

    def __hash__(self):
        return hash(('genes', self.print_genes()))

    def print_genes(self):
        return_genes = ""
        for i in range(0, len(self.genes)):
            return_genes += str(self.genes[i].included)#+","
        return return_genes

    def print_genes_indexes(self):
        return_genes = ""
        for i in range(0, len(self.genes)):
            if self.genes[i].included:
                return_genes += str(i+1)+","
            else:
                return_genes += "-"+","
        return return_genes

    def print_genes_array(self):
        return_genes = []
        for i in range(0, len(self.genes)):
            return_genes.append(self.genes[i].included)
        return return_genes