import random
from typing import List
from datasets.Dataset import Dataset
import numpy as np
import json
import math


def random_dataset_generator(num_pbis: int = 20, num_stakeholders: int = 5, percentage_dependencies: float = 0.45,
                             range_pbi_costs: List[int] = None, total_pbi_costs: int = None, range_stakeholder_importances: List[int] = None,
                             range_stakeholder_pbis_priorities: List[int] = None, name: str = "random") -> Dataset:

    # default values for lists
    default_range_pbi_costs = [1, 1, 2, 3, 5, 8, 13, 21, 34]  # Fibonacci
    # 1=no importance; 5=highest importance
    default_range_stakeholder_importances = [1, 2, 3, 4, 5]
    default_range_stakeholder_pbis_priorities = [1, 2, 3, 4, 5]

    # override values:
    range_pbi_costs = default_range_pbi_costs if range_pbi_costs is None else range_pbi_costs
    range_stakeholder_importances = default_range_stakeholder_importances if range_stakeholder_importances is None else range_stakeholder_importances
    range_stakeholder_pbis_priorities = default_range_stakeholder_pbis_priorities if range_stakeholder_pbis_priorities is None else range_stakeholder_pbis_priorities

    if num_pbis <= 0 or num_stakeholders <= 0 or percentage_dependencies < 0:
        raise Exception(
            "Parameters num_pbis, num_stakeholders, num_dependencies must be positive integers")
    if total_pbi_costs is not None and total_pbi_costs <= 0:
        raise Exception(
            "Total pbi cost must be positive")

    output_file: str = f"datasets/{name}.json"

    # if given the total sum cost of all pbis->generate randomly the costs using aux function
    if total_pbi_costs is not None:
        pbi_costs = _constrained_sum_sample_pos(num_pbis, total_pbi_costs)
    else:  # generate random pbi costs array
        pbi_costs = np.random.choice(range_pbi_costs, size=num_pbis)

    # generate random stakeholder importances array
    stakeholder_importances = np.random.choice(
        range_stakeholder_importances, size=num_stakeholders)

    # generate random array of priorities for all pbis for each stakeholder
    stakeholder_pbis_priorities = []
    for _ in range(num_stakeholders):
        priorities = np.random.choice(
            range_stakeholder_pbis_priorities, size=num_pbis)
        stakeholder_pbis_priorities.append(priorities.tolist())

    # calculate amount of dependencies given the num of pbis
    num_dependencies = math.floor(num_pbis*percentage_dependencies)

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
        "pbis_cost": pbi_costs.tolist(),
        "stakeholders_importances": stakeholder_importances.tolist(),
        "stakeholders_pbis_priorities": stakeholder_pbis_priorities,
        "dependencies": pbi_dependencies.tolist(),
        "_len_pbis_cost": len(pbi_costs),
        "_len_stakeholders_importances": len(stakeholder_importances),
        "_len_stakeholders_pbis_priorities": len(stakeholder_pbis_priorities),
        "_len_dependencies": len([x for x in pbi_dependencies if x is not None])
    }

    # store data in json file
    with open(output_file, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    # return dataset instance
    dataset = Dataset("random", source_file=output_file)
    return dataset


def _constrained_sum_sample_pos(num_pbis: int, total_sum_costs: int) -> np.ndarray:
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total_sum_costs), num_pbis - 1))
    return np.array([a - b for a, b in zip(dividers + [total_sum_costs], [0] + dividers)])
