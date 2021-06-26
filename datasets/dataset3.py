#DATASET 1
from algorithms.GRASP.Dataset import Dataset
from models.gen import Gen


def generate_dataset3_genes():
  dataset=Dataset("3")
  genes=[]
  for i in range(len(dataset.pbis_cost_scaled)):
    new_gen=Gen(None,dataset.pbis_satisfaction_scaled[i],dataset.pbis_cost_scaled[i])
    genes.append(new_gen)

  return genes