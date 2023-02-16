from typing import Dict


def combine_dataset_requirements(json_data: dict) -> dict:
    combined_requirements = dict()
    # for each requirement, check its dependencies
    for req_index in range(len(json_data["dependencies"])):
        req_dependencies = json_data["dependencies"][req_index]
        if req_dependencies is None:
            continue

        # if has dependent requirements, for each one check if those have any inverse dependency to the current requirement
        affected_requirements = set()  # {req_index}
        for other_dependency_index in req_dependencies:
            if is_requirement_affected_by_dependency(req_index, other_dependency_index, json_data):
                affected_requirements.add(other_dependency_index)
                # if so, also check all bidirectional dependencies of the other requirement (recursively)
                # affected_requirements.update(check_all_dependencies_of_requirement(other_dependency_index, json_data))

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
    for index in range(len(json_data["pbis_cost"])):

        if index not in list(set().union(*final_requirements_combined)):
            new_data["pbis_cost"].append(json_data["pbis_cost"][index])
            for stakeholder_index in range(len(json_data["stakeholders_importances"])):
                new_data["stakeholders_pbis_priorities"][stakeholder_index].append(
                    json_data["stakeholders_pbis_priorities"][stakeholder_index][index])
            new_index = len(new_data["pbis_cost"]) - 1
            new_indexes[index] = new_index

    for index in range(len(json_data["dependencies"])):
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
        new_data["dependencies"].append(None)

    return new_data


def is_requirement_affected_by_dependency(req_index: int, other_dependency_index: int, json_data: Dict) -> bool:
    # find if there is a dependency between the two requirements
    return req_index in json_data["dependencies"][other_dependency_index]


def check_all_dependencies_of_requirement(req_index: int, json_data: Dict) -> set:
    # for each requirement, check if it is affected by the current requirement
    dependent_requirements = set()
    for other_requirement_index in range(len(json_data["dependencies"])):
        if other_requirement_index != req_index:
            if is_requirement_affected_by_dependency(req_index, other_requirement_index, json_data):
                dependent_requirements.add(other_requirement_index)
                # if so, also check all bidirectional dependencies of the other requirement (recursively)
                dependent_requirements.update(
                    check_all_dependencies_of_requirement(other_requirement_index, json_data))

    return dependent_requirements
