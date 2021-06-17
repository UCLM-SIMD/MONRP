#DATASET 1
from algorithms.GRASP.Dataset import Dataset
from models.gen import Gen


def generate_dataset1_genes():
  ####importancias de stakeholders
  stakeholders_importances=[1, 4, 2, 3, 4]
  stakeholders=[]
  for i in range(0, len(stakeholders_importances)):
    stakeholders.append({
        "idusuario": i+1,
        "importance": stakeholders_importances[i]
        })
  ####estimaciones de pbis--------------------------------------------------------
  pbis_estimaciones=[1, 4, 2, 3, 4, 7, 10, 2, 1, 3, 2, 5, 8, 2, 1, 4, 10, 4, 8, 4]
  pbis=[]
  for i in range(0, len(pbis_estimaciones)):
    pbis.append({
        "idpbi":i+1,
        "estimacion":pbis_estimaciones[i]
    })

  '''
  for i in range(0, len(pbis_estimaciones)):
    print(pbis[i])
  '''
  ####valores de pbis para cada stakeholder---------------------------------------
  stakeholders_pbis_priorities=[]
  priorities1=[4 ,2 ,1 ,2 ,5 ,5 ,2 ,4 ,4 ,4 ,2 ,3 ,4 ,2 ,4 ,4 ,4 ,1 ,3 ,2]
  priorities2=[4 ,4 ,2 ,2 ,4 ,5 ,1 ,4 ,4 ,5 ,2 ,3 ,2 ,4 ,4 ,2 ,3 ,2 ,3 ,1]
  priorities3=[ 5, 3, 3, 3, 4, 5, 2, 4, 4, 4, 2, 4, 1, 5, 4, 1, 2, 3, 3, 2]
  priorities4=[ 4, 5, 2, 3, 3, 4, 2, 4, 2, 3, 5, 2, 3, 2, 4, 3, 5, 4, 3, 2]
  priorities5=[ 5, 4, 2, 4, 5, 4, 2, 4, 5, 2, 4, 5, 3, 4, 4, 1, 1, 2, 4, 1]
  stakeholders_pbis_priorities.append(priorities1)
  stakeholders_pbis_priorities.append(priorities2)
  stakeholders_pbis_priorities.append(priorities3)
  stakeholders_pbis_priorities.append(priorities4)
  stakeholders_pbis_priorities.append(priorities5)
  #print(stakeholders_pbis_priorities)

  ####creacion de genes:----------------------------------------------------------
  genes=[]
  for i in range(0,len(pbis)):
    value=0
    for j in range(0,len(stakeholders)):
      importance=stakeholders[j]["importance"]
      stakeholder_priority=stakeholders_pbis_priorities[j][i]
      value+=importance*stakeholder_priority
    value/=len(stakeholders)
    new_gen=Gen(pbis[i]["idpbi"],value,pbis[i]["estimacion"])
    #print(new_gen)
    genes.append(new_gen)

  dataset=Dataset("1")
  genes=[]
  for i in range(len(dataset.pbis_cost_scaled)):
    new_gen=Gen(pbis[i]["idpbi"],dataset.pbis_satisfaction_scaled[i],dataset.pbis_cost_scaled[i])
    genes.append(new_gen)

  return genes