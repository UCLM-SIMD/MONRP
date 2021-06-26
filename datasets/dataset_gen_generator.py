#DATASET 1
from algorithms.GRASP.Dataset import Dataset
from models.gen import Gen


def generate_dataset_genes(dataset_name):
  dataset=Dataset(dataset_name)
  genes=[]
  for i in range(len(dataset.pbis_cost_scaled)):
    new_gen=Gen(None,dataset.pbis_satisfaction_scaled[i],dataset.pbis_cost_scaled[i])
    genes.append(new_gen)

  return genes