from datasets.Dataset import Dataset
import numpy as np
import json
import math


def random_dataset_generator(num_pbis: int = None, num_stakeholders: int = None, num_dependencies: int = None,
                             output_file: str = "datasets/random.json", scale: str = "s1") -> Dataset:
    # scales:
    if scale == "s1":
        default_num_pbis = 40
        default_num_stakeholders = 15
    elif scale == "s2":
        default_num_pbis = 80
        default_num_stakeholders = 50
    elif scale == "s3":
        default_num_pbis = 140
        default_num_stakeholders = 100
    else:
        raise Exception("Scale does not exist")

    # define default num of dependencies if not given -> ~45%
    default_num_dependencies = math.floor(default_num_pbis*0.45)

    # default parameters:
    num_pbis = default_num_pbis if num_pbis is None else num_pbis
    num_stakeholders = default_num_stakeholders if num_stakeholders is None else num_stakeholders
    num_dependencies = default_num_stakeholders if num_dependencies is None else num_dependencies

    if num_pbis <= 0 or num_stakeholders <= 0 or num_dependencies <= 0:
        raise Exception(
            "Parameters num_pbis, num_stakeholders, num_dependencies must be positive integers")

    # default ranges:
    min_pbi_cost = 1
    max_pbi_cost = 40

    min_stakeholder_importance = 1
    max_stakeholder_importance = 5

    min_stakeholder_priorities = 1
    max_stakeholder_priorities = 5

    # generate random pbi costs array
    pbi_costs = np.random.randint(
        min_pbi_cost, (max_pbi_cost+1), size=num_pbis)

    # generate random stakeholder importances array
    stakeholder_importances = np.random.randint(
        min_stakeholder_importance, (max_stakeholder_importance+1), size=num_stakeholders)

    # generate random array of priorities for all pbis for each stakeholder
    stakeholder_priorities = []
    for _ in range(num_stakeholders):
        priorities = np.random.randint(
            min_stakeholder_priorities, (max_stakeholder_priorities+1), size=num_pbis)
        stakeholder_priorities.append(priorities.tolist())

    # generate valid dependencies until max and store them in a dict
    done_dependencies = {}
    counter_dependencies = 0
    while(counter_dependencies < num_dependencies):
        random_pbi1 = np.random.randint(0, num_pbis)
        random_pbi2 = np.random.randint(0, num_pbis)
        if (random_pbi1 == random_pbi2):
            continue
        if(random_pbi1 in done_dependencies and random_pbi2 in done_dependencies[random_pbi1]):
            continue
        done_dependencies.setdefault(random_pbi1, []).append(random_pbi2)
        counter_dependencies += 1

    # transform dict values to array of None or nested array elements
    pbi_dependencies = np.empty(num_pbis, dtype=object)
    for key, value in done_dependencies.items():
        pbi_dependencies[key] = []
        pbi_dependencies[key] = np.append(pbi_dependencies[key], value)
        pbi_dependencies[key] = pbi_dependencies[key].tolist()
        for x in range(len(pbi_dependencies[key])):
            pbi_dependencies[key][x] = int(pbi_dependencies[key][x])

    # return format
    json_data = {
        "costs": pbi_costs.tolist(),
        "importances": stakeholder_importances.tolist(),
        "priorities": stakeholder_priorities,
        "dependencies": pbi_dependencies.tolist(),
    }

    # store data in json file
    with open(output_file, "w") as json_file:
        json.dump(json_data, json_file)

    # return dataset instance
    dataset = Dataset("random", source_file=output_file)
    return dataset
