import math


#HYPERVOLUME------------------------------------------------------------------
def calculate_hypervolume(population):
  #obtener minimos y maximos de cada objetivo
  score_min=float('inf')
  score_max=0
  cost_min=float('inf')
  cost_max=0
  for ind in population:
    if ind.score>score_max:
      score_max=ind.score
    if ind.score<score_min:
      score_min=ind.score

    if ind.cost>cost_max:
      cost_max=ind.cost
    if ind.cost<cost_min:
      cost_min=ind.cost

  #calcular hypervolume
  score_diff=score_max-score_min
  cost_diff=cost_max-cost_min
  hypervolume=score_diff*cost_diff
  return hypervolume

#SPREAD------------------------------------------------------------------
def eudis(v1, v2):
  dist = [(a - b)**2 for a, b in zip(v1, v2)]
  dist = math.sqrt(sum(dist))
  return dist

def calculate_spread(population):
  MAX_SCORE=25 #max_importancia_Stakeholder * max_prioridad_pbi_para_Stakeholder
  MAX_COST=40 #max estimacion de pbi
  df=None
  dl=None
  davg=None
  sum_dist=None
  N=len(population)
  spread=None
  #ordenar de menor a mayor coste
  population.population.sort(key=lambda x: x.cost)
  #for p in population:
   # print(p)
  #obtener first_solution=menor coste y last_solution=mayor coste
  first_solution=population.get(0)
  last_solution=population.get(len(population)-1)

  #obtener first_extreme=[score=0 (worst),cost=0 (best)] y last_extreme=[score=MAX_SCORE (best),cost=MAX_COST (worst)]
  first_extreme=[0,0]
  last_extreme=[MAX_SCORE,MAX_COST]

  df=eudis( [first_solution.score,first_solution.cost ] , first_extreme )
  dl=eudis( [last_solution.score,last_solution.cost ] , last_extreme )

  #calcular media de todas las distancias entre puntos
  davg=0
  dist_count=0
  for i in range(0,len(population)):
    for j in range(0,len(population)):
      #no calcular distancia de un punto a si mismo
      if i!=j:
        dist_count+=1
        davg+=eudis( [population.get(i).score,population.get(i).cost ],
               [population.get(j).score,population.get(j).cost ] )
  #media=distancia total / numero de distancias
  davg/=dist_count

  #calcular sumatorio(i=1->N-1) |di-davg|
  sum_dist=0
  for i in range(0,len(population)-1):
    di=eudis( [population.get(i).score,population.get(i).cost ],
               [population.get(i+1).score,population.get(i+1).cost ] )
    sum_dist+=abs(di-davg)

  #formula spread
  spread=(df+dl+sum_dist)/(df+dl+(N-1)*davg)
  return spread

# FAST NONDOMINATED SORT
def fast_nondominated_sort(population):
  population.fronts = [[]]
  for individual in population:
    individual.domination_count = 0
    individual.dominated_solutions = []
    for other_individual in population:
      if individual.dominates(other_individual):
          individual.dominated_solutions.append(other_individual)
      elif other_individual.dominates(individual):
          individual.domination_count += 1
    if individual.domination_count == 0:
      individual.rank = 0
      population.fronts[0].append(individual)
  i = 0
  while len(population.fronts[i]) > 0:
    temp = []
    for individual in population.fronts[i]:
      for other_individual in individual.dominated_solutions:
        other_individual.domination_count -= 1
        if other_individual.domination_count == 0:
          other_individual.rank = i+1
          temp.append(other_individual)
    i = i+1
    population.fronts.append(temp)