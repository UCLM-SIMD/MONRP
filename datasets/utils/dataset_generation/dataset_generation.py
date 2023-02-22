import math
import json
import numpy as np
import argparse
import random

from typing import List
import sys

from datasets.Dataset import Dataset


def random_dataset_generator(name: str = "random",  num_pbis: int = 20, num_stakeholders: int = 5, percentage_dependencies: float = 0.45,
                             range_pbi_costs: List[int] = None, total_pbi_costs: int = None, range_stakeholder_importances: List[int] = None,
                             range_stakeholder_pbis_priorities: List[int] = None, avg_len_dependencies: int = None) -> Dataset:
    """
    avg_len_dependencies: avg length of the list of of PBIs implied by a dependency p-->[PBIs]
    """
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
    while (counter_dependencies < num_dependencies):
        random_pbi1 = np.random.randint(0, num_pbis)

        if avg_len_dependencies is None:
            random_pbi2 = np.random.randint(0, num_pbis)
            if (random_pbi1 == random_pbi2):
                continue
            if (random_pbi1 in done_dependencies and random_pbi2 in done_dependencies[random_pbi1]):
                continue
            done_dependencies.setdefault(random_pbi1, []).append(random_pbi2)
        else:
            random_list_pbi2 = random.sample(range(num_pbis), np.random.randint(
                avg_len_dependencies-1, avg_len_dependencies+1))
            if random_pbi1 in done_dependencies:
                continue
            if random_pbi1 in random_list_pbi2:
                random_list_pbi2.remove(random_pbi1)
            done_dependencies.setdefault(
                random_pbi1, []).extend(random_list_pbi2)

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
    output_file: str = f"{name}.json"
    with open(output_file, "w", newline='\n') as json_file:
        json.dump(json_data, json_file, indent=4)

    # return dataset instance
    dataset = Dataset("random", source_file=output_file)
    return dataset


def _constrained_sum_sample_pos(num_pbis: int, total_sum_costs: int) -> np.ndarray:
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total_sum_costs), num_pbis - 1))
    return np.array([a - b for a, b in zip(dividers + [total_sum_costs], [0] + dividers)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a random dataset for the PBI prioritization problem")
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="Name of the dataset")
    parser.add_argument("-p", "--num_pbis", type=int, required=True,
                        help="Number of pbis")
    parser.add_argument("-s", "--num_stakeholders", type=int, required=True,
                        help="Number of stakeholders")
    parser.add_argument("-d", "--percentage_dependencies", type=float, required=True,
                        help="Percentage of dependencies")
    parser.add_argument("-t", "--total_pbi_costs", type=int,
                        help="Total pbi costs")
    parser.add_argument("-a", "--avg_len_dependencies", type=int,
                        help="Average length of dependencies")
    parser.add_argument("-rpc", "--range_pbi_costs", type=int, nargs='+',
                        help="Range of pbi costs as min and max values, both included: 0 5 --> [0, 1, 2 ,3 ,4 ,5]")
    parser.add_argument("-rsi", "--range_stakeholder_importances", type=int, nargs='+',
                        help="Range of stakeholder importances")
    parser.add_argument("-rspp", "--range_stakeholder_pbis_priorities", type=int, nargs='+',
                        help="Range of stakeholder pbis priorities")
    args = parser.parse_args()

    args.range_pbi_costs = range(
        args.range_pbi_costs[0], args.range_pbi_costs[1]) if args.range_pbi_costs is not None else None

    args.range_stakeholder_importances = range(
        args.range_stakeholder_importances[0], args.range_stakeholder_importances[1]) if args.range_stakeholder_importances is not None else None

    args.range_stakeholder_pbis_priorities = range(
        args.range_stakeholder_pbis_priorities[0], args.range_stakeholder_pbis_priorities[1]) if args.range_stakeholder_pbis_priorities is not None else None

    random_dataset_generator(args.name, args.num_pbis, args.num_stakeholders, args.percentage_dependencies, args.range_pbi_costs, args.total_pbi_costs,
                             args.range_stakeholder_importances, args.range_stakeholder_pbis_priorities, args.avg_len_dependencies)
    print(f"Dataset {args.name}.json generated successfully:")
    print(f" - Number of pbis: {args.num_pbis}")
    print(f" - Number of stakeholders: {args.num_stakeholders}")
    print(f" - Percentage of dependencies: {args.percentage_dependencies}")
