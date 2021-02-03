class Population:
  def __init__(self):
      self.population = []
      self.fronts = []
  def __len__(self):
      return len(self.population)
  def __iter__(self):
      return self.population.__iter__()
  def extend(self, new_individuals):
      self.population.extend(new_individuals)
  def append(self, new_individual):
      self.population.append(new_individual)
  def index(self, element):
    return self.population.index(element)
  def get(self, index):
      return self.population[index]
  def set(self, index, individual):
    self.population[index]=individual