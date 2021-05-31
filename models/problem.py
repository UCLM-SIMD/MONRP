from models.individual import Solution
from models.objective import Objective


class Problem:
  def __init__(self, genes,minimize_list):
    self.genes=genes
    self.objectives = []
    for elem in minimize_list:
        if elem =="MAX":
            self.objectives.append(Objective(None, minimize=False))
        elif elem == "MIN":
            self.objectives.append(Objective(None, minimize=True))

  # GENERATE INDIVIDUAL------------------------------------------------------------------
  def generate_individual(self, genes):
    individual=Solution(genes, self.objectives)
    individual.initRandom()
    return individual

  # GENERATE STARTING INDIVIDUAL------------------------------------------------------------------
  def generate_starting_individual(self):
    individual=Solution(self.genes, self.objectives)
    individual.initRandom()
    return individual

