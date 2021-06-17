#DATASET 1
from algorithms.GRASP.Dataset import Dataset
from models.gen import Gen


def generate_dataset2_genes():
  ####importancias de stakeholders
  stakeholders_importances=[1, 5, 3, 3, 1]
  stakeholders=[]
  for i in range(0, len(stakeholders_importances)):
    stakeholders.append({
        "idusuario": i+1,
        "importance": stakeholders_importances[i]
        })
  ####estimaciones de pbis--------------------------------------------------------
  pbis_estimaciones=[16,19,16,7,19,15,8,10,6,18,15,12,16,20,9,4,16,2,9,3,2,10,4,2,7,15,8,
  20,9,11,5,1,17,6,2,16,8,12,18,5,6,14,15,20,14,9,16,6,6,6,6,2,17,8,1,3,14,16,18,7,10,7,16,
  19,17,15,11,8,20,1,5,8,3,15,4,20,10,20,3,20,10,16,19,3,12,16,15,1,6,7,15,18,4,7,2,7,8,7,7,3]
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
  priorities1=[1,2,1,1,2,3,3,1,1,3,1,1,3,2,3,2,2,3,1,3,2,1,1,1,3,3,3,3,1,2,2,3,2,1,2,2,1
  ,3,3,2,2,2,3,1,1,1,2,2,3,3,3,3,1,3,2,1,3,1,3,1,2,2,3,3,1,3,1,3,2,3,1,3,2,3,1,1,2,3,3,1,
  2,1,3,1,2,2,2,1,3,2,2,3,1,1,1,2,1,3,1,1]
  priorities2=[3,2,1,2,1,2,1,2,2,1,2,3,3,2,1,3,2,3,3,1,3,3,3,2,3,1,2,2,3,3,1,3,2,2,1,2,3,
  2,3,3,3,3,1,1,3,2,2,2,1,3,3,3,1,2,2,3,3,2,1,1,1,3,2,3,1,2,1,2,3,1,1,3,1,3,2,1,3,3,1,2,1,
  2,1,2,2,1,3,2,2,2,3,2,2,3,2,2,1,3,1,1]
  priorities3=[1,1,1,2,1,1,1,3,2,2,3,3,3,1,3,1,2,2,3,3,2,1,2,3,2,3,3,1,3,3,3,2,1,2,2,1,1,3
  ,1,2,1,3,1,3,3,3,3,1,3,2,3,1,2,3,2,3,2,1,2,3,1,1,2,3,3,1,3,3,3,1,3,1,3,1,1,2,3,3,1,2,1,2
  ,3,2,3,1,2,2,3,3,3,3,2,1,1,2,3,3,2,3]
  priorities4=[3,2,2,1,3,1,3,2,3,2,3,2,1,3,2,3,2,1,3,3,1,1,1,2,3,3,2,1,1,1,1,2,2,2,3,2,2,3
  ,1,1,3,1,1,3,1,2,1,1,3,2,2,1,3,2,1,3,3,1,2,3,2,2,3,3,3,1,2,1,2,1,2,3,3,2,2,2,1,3,3,1,3,1
  ,2,2,2,1,1,1,3,1,1,3,3,1,2,1,2,3,1,3]
  priorities5=[1,2,3,1,3,1,2,3,1,1,2,2,3,1,2,1,1,1,1,3,1,1,3,3,3,2,2,3,2,3,1,1,3,3,2,2,1,1
  ,2,1,3,1,1,2,1,2,3,3,2,2,1,3,3,2,3,1,2,1,3,2,2,2,1,2,1,3,2,1,2,1,2,2,3,2,1,3,2,3,1,3,3,2,
  1,2,2,2,2,1,3,3,3,1,1,3,1,3,3,3,3,3]
  stakeholders_pbis_priorities.append(priorities1)
  stakeholders_pbis_priorities.append(priorities2)
  stakeholders_pbis_priorities.append(priorities3)
  stakeholders_pbis_priorities.append(priorities4)
  stakeholders_pbis_priorities.append(priorities5)
  #print(stakeholders_pbis_priorities)

  ####creacion de genes:----------------------------------------------------------
  #genes=[]
  #for i in range(0,len(pbis)):
  #  value=0
  #  for j in range(0,len(stakeholders)):
  #    importance=stakeholders[j]["importance"]
  #    stakeholder_priority=stakeholders_pbis_priorities[j][i]
  #    value+=importance*stakeholder_priority
  #  value/=len(stakeholders)
  #  new_gen=Gen(pbis[i]["idpbi"],value,pbis[i]["estimacion"])
  #  #print(new_gen)
  #  genes.append(new_gen)


  dataset=Dataset("2")
  genes=[]
  for i in range(len(dataset.pbis_cost_scaled)):
    new_gen=Gen(pbis[i]["idpbi"],dataset.pbis_satisfaction_scaled[i],dataset.pbis_cost_scaled[i])
    genes.append(new_gen)

  return genes