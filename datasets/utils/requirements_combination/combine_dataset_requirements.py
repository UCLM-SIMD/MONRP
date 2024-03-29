import json
import sys
from typing import Dict


def combine_dataset_requirements(json_data: dict, log_info: bool = False) -> dict:
    """
    Combines requirements that have interaction dependencies of type <combination> between them. 

    For example, given a set of requirements { r1, r2, r3, r4 } and the following combination dependency:
    r1 <--> r2 (coded as r1 -> r2; r2 -> r1); the resulting requirements set will be { r1+r2, r3, r4 }, where r1+r2 is the combination of r1 and r2.
    """
    if "dependencies" not in json_data or json_data["dependencies"] is None:
        raise Exception("No dependencies found in dataset")
    if "pbis_cost" not in json_data or json_data["pbis_cost"] is None:
        raise Exception("No pbis_cost found in dataset")
    if "stakeholders_importances" not in json_data or json_data["stakeholders_importances"] is None:
        raise Exception("No stakeholders_importances found in dataset")
    if "stakeholders_pbis_priorities" not in json_data or json_data["stakeholders_pbis_priorities"] is None:
        raise Exception("No stakeholders_pbis_priorities found in dataset")

    combined_requirements = dict()
    # for each requirement, check its dependencies
    for req_index in range(len(json_data["dependencies"])):
        req_dependencies = json_data["dependencies"][req_index]
        if req_dependencies is None:
            continue

        # if has dependent requirements, for each one check if those have any inverse dependency to the current requirement
        affected_requirements = set()
        for other_dependency_index in req_dependencies:
            if is_requirement_affected_by_dependency(req_index, other_dependency_index, json_data):
                affected_requirements.add(other_dependency_index)

        if affected_requirements:
            combined_requirements[req_index] = affected_requirements

    # for each requirement and its combined requirement dependencies
    combined_requirements_sets = []
    for req_index, dependent_requirements in combined_requirements.items():
        # create a set containing the combination of requirements
        combined_requirements_set = {req_index}
        combined_requirements_set.update(dependent_requirements)

        # if the requirement is already assigned to a requirement combination, update the existing set
        existing_requirements_set = [
            req_set for req_set in combined_requirements_sets if req_index in req_set]
        if any(existing_requirements_set):
            existing_requirements_set[0].update(combined_requirements_set)
        # else create a new set
        else:
            combined_requirements_sets.append(combined_requirements_set)

    # combine all sets
    final_requirements_combined = []
    visited_sets = set()
    for index, current_set in enumerate(combined_requirements_sets):
        if index in visited_sets:
            continue
        visited_sets.add(index)
        new_combination = set(current_set)
        # for each other non-visited set, if they have some requirement in common, merge them and mark as visited
        for other_index, other_set in enumerate(combined_requirements_sets):
            if other_index == index or other_index in visited_sets:
                continue
            if any([req for req in other_set if req in current_set]):
                visited_sets.add(other_index)
                new_combination.update(other_set)
        final_requirements_combined.append(new_combination)

    new_data = {
        "pbis_cost": [],
        "stakeholders_importances": json_data["stakeholders_importances"],
        "stakeholders_pbis_priorities": [[] for _ in json_data["stakeholders_importances"]],
        "dependencies": []
    }

    new_indexes = {}
    # calculate new indexes and reindex requirements not combined
    for index in range(len(json_data["pbis_cost"])):
        if index not in list(set().union(*final_requirements_combined)):
            new_data["pbis_cost"].append(json_data["pbis_cost"][index])
            for stakeholder_index in range(len(json_data["stakeholders_importances"])):
                new_data["stakeholders_pbis_priorities"][stakeholder_index].append(
                    json_data["stakeholders_pbis_priorities"][stakeholder_index][index])
            new_index = len(new_data["pbis_cost"]) - 1
            new_indexes[index] = new_index

    # calculate new indexes of requirement combinations
    for index in range(len(json_data["pbis_cost"])):
        if index in list(set().union(*final_requirements_combined)):
            # map each combination of requirements to a new requirement_id
            final_requirements_combined_index = next(
                (i for i, v in enumerate(final_requirements_combined) if index in v), None)
            new_index = len(new_data["pbis_cost"]) + \
                final_requirements_combined_index
            # get final_requirements_combined index where the current index is located
            new_indexes[index] = new_index

    # for each noncombined requirement, add its former dependencies, translating them to the new indexes
    for index in range(len(json_data["pbis_cost"])):
        if index not in list(set().union(*final_requirements_combined)):
            new_data["dependencies"].append(json_data["dependencies"][index])
            if new_data["dependencies"][-1] is not None:
                new_data["dependencies"][-1] = [new_indexes[dep]
                                                for dep in new_data["dependencies"][-1]]

    # for each set of combined requirements, calculate the combined cost and importance and add them to the new dataset
    for combined_requirements_set in final_requirements_combined:
        # calculate cost as mean of all requirements costs
        new_cost = sum([json_data["pbis_cost"][index]
                        for index in combined_requirements_set])

        # calculate priority as max of all requirements priorities for all stakeholders
        new_priorities = [
            max([json_data["stakeholders_pbis_priorities"][stakeholder_index][index] for index in
                 combined_requirements_set])
            for stakeholder_index in range(len(json_data["stakeholders_importances"]))]
        new_data["pbis_cost"].append(new_cost)
        for stakeholder_index in range(len(json_data["stakeholders_importances"])):
            new_data["stakeholders_pbis_priorities"][stakeholder_index].append(
                new_priorities[stakeholder_index])

        comb_requirements_dependencies = []
        for index in combined_requirements_set:
            comb_requirements_dependencies.append(
                json_data["dependencies"][index])
            if comb_requirements_dependencies[-1]:
                comb_requirements_dependencies[-1] = [new_indexes[dep] for dep in comb_requirements_dependencies[-1]
                                                      if dep not in combined_requirements_set]
        # flatten comb_requirements_dependencies
        comb_requirements_dependencies = [
            item for sublist in comb_requirements_dependencies for item in sublist]
        new_data["dependencies"].append(
            list(set(comb_requirements_dependencies)))
        if not new_data["dependencies"][-1]:
            new_data["dependencies"][-1] = None

    if log_info:
        print("Previous number of requirements: {}".format(
            len(json_data["pbis_cost"])))
        print("New number of requirements: {}".format(
            len(new_data["pbis_cost"])))
        print("Number of combination dependencies found: {}".format(
            len(final_requirements_combined)))
        print(
            f"Requirement combinations: {[', '.join([f'r_{index}' for index in combination]) for combination in final_requirements_combined]}")

    return new_data


def is_requirement_affected_by_dependency(req_index: int, other_dependency_index: int, json_data: Dict) -> bool:
    # find if there is a dependency between the two requirements
    return json_data["dependencies"][other_dependency_index] is not None and \
        req_index in json_data["dependencies"][other_dependency_index]


if __name__ == "__main__":
    if len(sys.argv) == 2:
        json_filename = sys.argv[1]
    else:
        raise Exception(
            "Invalid number of arguments. Usage: python3 combine_dataset_requirements.py <input_json_filename>")

    with open(json_filename, "r") as json_file:
        json_data = json.load(json_file)

    new_data = combine_dataset_requirements(json_data, log_info=True)

    output_filename = json_filename.replace(".json", "_combined.json")
    with open(output_filename, "w") as json_file:
        json.dump(new_data, json_file)

    print("Dataset requirements successfully combined. Output file: {}".format(
        output_filename))
