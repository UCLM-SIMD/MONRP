from datasets.dataset_p1 import dataset_p1
from datasets.dataset_p2 import dataset_p2
from datasets.dataset_s1 import dataset_s1
from datasets.dataset_s2 import dataset_s2
from datasets.dataset_s3 import dataset_s3
from datasets.dataset_test import dataset_test
from datasets.normalize_dataset import normalize_dataset
import numpy as np


class Dataset:
    """
    Class used to load datasets as published in
    del Sagrado, José del Águila, Isabel M. Orellana, Francisco J.
    Multi-objective ant colony optimization for requirements selection.
    Empirical Software Engineering. Vol. 20(3). 2015.

    Attributes
    -----------
    id :  string
        1 or 2, currently.

    num_pbis : integer
        just used to represent the number of pbis or candidates to be selected

    pbis_cost : numpy ndarray
        Cost of each product backlog item

    stakeholders_importances: numpy ndarray
        Importance or weight assigned to each stakeholder

    stakeholders_pbis_priorities: numpy ndarray, shape is (len(self.stakeholders_importances),self.num_pbis)
        Importance of each pbi for each stakeholder

    pbis_satisfaction: numpy ndarray, shape is (self.num_pbis)
        satisfaction[i] is the sum of stakeholders_pbis_priorities(_,i) each value weighted by self.stakeholders_importances[j]

    pbis_satisfaction_scaled: numpy ndarray, shape is (self.num_pbis)
        scaled version of self.pbis_satisfaction

    pbis_cost_scaled: numpy ndarray, shape is (self.num_pbis)
        scaled version of self.pbis_cost

    pbis_score: numpy ndarray, shape is (self.num_pbis)
        mixture of each pbi satisfaction and cost.


    """

    def __init__(self, dataset: str = "test"):
        """
        :param
        dataset: string: 1 or 2 (at the moment)
            if dataset not in [1,2], all attributes are set to None.
        """

        self.id = dataset

        if dataset == "test":  # 2 clientes 5 reqs 2 ints: 1-2-3-4-5; 1->2; 4->1
            self.pbis_cost, self.num_pbis, self.stakeholders_importances, self.stakeholders_pbis_priorities, self.dependencies = dataset_test()
        elif dataset == "1":  # 5 clientes 20 reqs 10 ints
            self.pbis_cost, self.num_pbis, self.stakeholders_importances, self.stakeholders_pbis_priorities, self.dependencies = dataset_p1()
        elif dataset == "2":  # 5 clientes 100 reqs 44ints
            self.pbis_cost, self.num_pbis, self.stakeholders_importances, self.stakeholders_pbis_priorities, self.dependencies = dataset_p2()
        elif dataset == "s1":  # 15 customers 40 reqs
            self.pbis_cost, self.num_pbis, self.stakeholders_importances, self.stakeholders_pbis_priorities, self.dependencies = dataset_s1()
        elif dataset == "s2":  # 50 customers 80 reqs
            self.pbis_cost, self.num_pbis, self.stakeholders_importances, self.stakeholders_pbis_priorities, self.dependencies = dataset_s2()
        elif dataset == "s3":  # 100 customers 140 reqs
            self.pbis_cost, self.num_pbis, self.stakeholders_importances, self.stakeholders_pbis_priorities, self.dependencies = dataset_s3()
        else:
            raise Exception("Sorry, dataset with id=", id, " not found.")

        #self.num_pbis: int = len(self.pbis_cost)
        # self.pbis_satisfaction: np.ndarray = self.stakeholders_importances.dot(
        #    self.stakeholders_pbis_priorities)

        # now two escalation follows, based on
        # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
        # scale pbis cost in range [0-1]
        # margin = 1 / self.num_pbis  # used to avoid zeros
        #diff = np.max(self.pbis_cost) - np.min(self.pbis_cost)
        # self.pbis_cost_scaled:np.ndarray = (
        #    self.pbis_cost - np.min(self.pbis_cost) + margin) / (diff + margin)

        # scale pbis satisfaction in range[0-1]
        #diff = np.max(self.pbis_satisfaction) - np.min(self.pbis_satisfaction)
        # self.pbis_satisfaction_scaled:np.ndarray = (
        #    self.pbis_satisfaction - np.min(self.pbis_satisfaction) + margin) / (diff + margin)

        #self.pbis_score = self.pbis_satisfaction_scaled / self.pbis_cost_scaled

        # each pbi score is computed from the scaled versions of pbi satisfaction and cost
        self.pbis_cost_scaled, self.pbis_satisfaction_scaled, self.pbis_score = normalize_dataset(
            self.pbis_cost, self.stakeholders_importances, self.stakeholders_pbis_priorities)

        # simplify dependencies:
        if (self.dependencies is not None):
            self.calculate_dependencies()

    def calculate_dependencies(self) -> None:
        self.new_dependencies = {}
        # dependency = index_dependency+1 (starts from 1)
        for dep in range(1, len(self.dependencies)+1):
            if self.dependencies[dep-1] is not None:
                # if req has dependencies -> add them and launch aux fun
                for dep2 in self.dependencies[dep-1]:
                    self.new_dependencies.setdefault(dep, []).append(dep2)
                    self.aux_dependencies(dep, dep2)

        # store new dependencies non repeatedly:
        self.dependencies = np.empty(len(self.dependencies), dtype=object)
        for i in range(1, len(self.dependencies)+1):
            if i not in self.new_dependencies:
                self.dependencies[i-1] = None
            else:
                self.dependencies[i -
                                  1] = list(dict.fromkeys(self.new_dependencies[i]))

    def aux_dependencies(self, parent: int, child: int) -> None:
        # if no dependencies in child -> stop
        if self.dependencies[child-1] is None:
            return
        # for each dependency in child -> if it is already the parent or contained in parent -> stop
        for d in self.dependencies[child-1]:
            if (d == parent) or (d in self.new_dependencies[parent]):
                continue
            # if not -> add new dependency to parent list and recursively launch aux fun
            self.new_dependencies.setdefault(parent, []).append(d)
            self.aux_dependencies(parent, d)
