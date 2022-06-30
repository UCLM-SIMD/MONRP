from pathlib import Path
from typing import Dict
import numpy as np
import json
import uuid


class Dataset:
    """
    Class used to load datasets as published in
    del Sagrado, José del Águila, Isabel M. Orellana, Francisco J.
    Multi-objective ant colony optimization for requirements selection.
    Empirical Software Engineering. Vol. 20(3). 2015.
    """

    def __init__(self, dataset: str = "test", source_file: str = None, source_dict: Dict = None):
        """Loads dataset vectors depending on the dataset name.
        """
        if source_file:
            with open(source_file) as json_file:
                # use filename as dataset id
                self.id = Path(source_file).stem
                json_data = json.load(json_file)
                self.load_from_dict(json_data)
        elif source_dict:
            self.id = uuid.uuid4().hex
            self.load_from_dict(source_dict)
        else:  # if not source file->search dataset json in datasets folder
            self.id = dataset
            with open("datasets/"+dataset+".json") as json_file:
                json_data = json.load(json_file)
                self.load_from_dict(json_data)

        # normalize values calculating scaled satisfactions, costs and scores
        self.normalize()

        # simplify dependencies:
        if self.dependencies is not None:
            self.list_of_sons = self.dependencies.copy()  # needed in feda_algorithm.py
            self.calculate_dependencies()

    def load_from_dict(self, source_dict: Dict) -> None:
        self.pbis_cost = np.array(source_dict["pbis_cost"]).astype(int)
        self.num_pbis = len(self.pbis_cost)
        self.stakeholders_importances = np.array(
            source_dict["stakeholders_importances"]).astype(int)
        self.stakeholders_pbis_priorities = np.array(
            source_dict["stakeholders_pbis_priorities"]).astype(int)
        if "dependencies" in source_dict:
            self.dependencies = np.array(
                source_dict["dependencies"], dtype=object)
            for x in range(len(self.dependencies)):
                if self.dependencies[x] is None:
                    continue
                for y in range(len(self.dependencies[x])):
                    self.dependencies[x][y] = int(self.dependencies[x][y])

        else:
            self.dependencies = None

    def calculate_dependencies(self) -> None:
        """Given the list of dependencies, recursively stores dependencies of requirements,
        saving in each requirements index all the requirements that have to be included to satisfy the dependency restrictions.
        """
        self.new_dependencies = {}
        # dependency = index_dependency+1 (starts from 1)
        for dep in range(len(self.dependencies)):
            if self.dependencies[dep] is not None:
                # if req has dependencies -> add them and launch aux fun
                for dep2 in self.dependencies[dep]:
                    self.new_dependencies.setdefault(dep, []).append(dep2)
                    self.aux_dependencies(dep, dep2)

        # store new dependencies non repeatedly:
        self.dependencies = np.empty(len(self.dependencies), dtype=object)
        for i in range(len(self.dependencies)):
            if i not in self.new_dependencies:
                self.dependencies[i] = None
            else:
                self.dependencies[i] = list(
                    dict.fromkeys(self.new_dependencies[i]))

    def aux_dependencies(self, parent: int, child: int) -> None:
        # if no dependencies in child -> stop
        if self.dependencies[child] is None:
            return
        # for each dependency in child -> if it is already the parent or contained in parent -> stop
        for d in self.dependencies[child]:
            if (d == parent) or (d in self.new_dependencies[parent]):
                continue
            # if not -> add new dependency to parent list and recursively launch aux fun
            self.new_dependencies.setdefault(parent, []).append(d)
            self.aux_dependencies(parent, d)

    def normalize(self) -> None:
        """Given the costs, importances and priorities, this method calculates the total satisfaction and score, and scales cost and
        satisfaction using min-max normalization
        """
        num_pbis = len(self.pbis_cost)

        self.pbis_satisfaction = self.stakeholders_importances.dot(
            self.stakeholders_pbis_priorities)

        # now two escalation follows, based on
        # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
        # scale pbis cost in range [0-1]
        margin = 1 / num_pbis  # used to avoid zeros
        diff = np.sum(self.pbis_cost) - np.min(self.pbis_cost)
        self.pbis_cost_scaled = (self.pbis_cost - np.min(self.pbis_cost) +
                                 margin) / (diff + margin)
        
        diff = np.sum(self.pbis_satisfaction) - np.min(self.pbis_satisfaction)
        self.pbis_satisfaction_scaled = (
            self.pbis_satisfaction - np.min(self.pbis_satisfaction) + margin) / (diff + margin)


        self.pbis_cost_scaled = self.pbis_cost_scaled / sum(self.pbis_cost_scaled)
        self.pbis_satisfaction_scaled = self.pbis_satisfaction_scaled / sum(self.pbis_satisfaction_scaled)
        #print(np.sum(self.pbis_cost_scaled))
        #print(np.sum(self.pbis_satisfaction_scaled))


        # each pbi score is computed from the scaled versions of pbi satisfaction and cost
        self.pbis_score = self.pbis_satisfaction_scaled / self.pbis_cost_scaled
