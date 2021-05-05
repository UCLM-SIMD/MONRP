import numpy as np


class Dataset:
    """
    Class used to load datasets as published in
    del Sagrado, José del Águila, Isabel M. Orellana, Francisco J.
    Multi-objective ant colony optimization for requirements selection.
    Empirical Software Engineering. Vol. 20(3). 2015.

    Attributes
    -----------
    id :  integer
        1 or 2, currently.

    num_pbis : integer
        just used to represent the number of pbis or candidates to be selectd

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

    def __init__(self, dataset):
        """
        :param
        dataset: integer number: 1 or 2 (at the moment)
            if dataset not in [1,2], all attributes are set to None.
        """

        self.id = dataset

        if dataset == 1:
            self.pbis_cost = np.array([1, 4, 2, 3, 4, 7, 10, 2, 1, 3, 2, 5, 8, 2, 1, 4, 10, 4, 8, 4])
            self.stakeholders_importances = np.array([1, 4, 2, 3, 4])
            self.stakeholders_pbis_priorities = np.array([4, 2, 1, 2, 5, 5, 2, 4, 4, 4, 2, 3, 4, 2, 4, 4, 4, 1, 3, 2])
            self.stakeholders_pbis_priorities = np.vstack((self.stakeholders_pbis_priorities,
                                                           [4, 4, 2, 2, 4, 5, 1, 4, 4, 5, 2, 3, 2, 4, 4, 2, 3, 2, 3,
                                                            1]))
            self.stakeholders_pbis_priorities = np.vstack((self.stakeholders_pbis_priorities,
                                                           [5, 3, 3, 3, 4, 5, 2, 4, 4, 4, 2, 4, 1, 5, 4, 1, 2, 3, 3,
                                                            2]))
            self.stakeholders_pbis_priorities = np.vstack((self.stakeholders_pbis_priorities,
                                                           [4, 5, 2, 3, 3, 4, 2, 4, 2, 3, 5, 2, 3, 2, 4, 3, 5, 4, 3,
                                                            2]))
            self.stakeholders_pbis_priorities = np.vstack((self.stakeholders_pbis_priorities,
                                                           [5, 4, 2, 4, 5, 4, 2, 4, 5, 2, 4, 5, 3, 4, 4, 1, 1, 2, 4,
                                                            1]))

        elif dataset == 2:
            self.pbis_cost = np.array([16, 19, 16, 7, 19, 15, 8, 10, 6, 18, 15, 12, 16, 20, 9, 4, 16, 2, 9, 3,
                                       2, 10, 4, 2, 7, 15, 8, 20, 9, 11, 5, 1, 17, 6, 2, 16, 8, 12, 18, 5, 6,
                                       14, 15, 20, 14, 9, 16, 6, 6, 6, 6, 2, 17, 8, 1, 3, 14, 16, 18, 7, 10, 7,
                                       16, 19, 17, 15, 11, 8, 20, 1, 5, 8, 3, 15, 4, 20, 10, 20, 3, 20, 10, 16,
                                       19, 3, 12, 16, 15, 1, 6, 7, 15, 18, 4, 7, 2, 7, 8, 7, 7, 3])
            self.num_pbis = len(self.pbis_cost)
            self.stakeholders_importances = np.array([1, 4, 2, 3, 4])

            self.stakeholders_pbis_priorities = np.array([1, 2, 1, 1, 2, 3, 3, 1, 1, 3, 1, 1, 3, 2, 3, 2, 2, 3, 1, 3,
                                                          2, 1, 1, 1, 3, 3, 3, 3, 1, 2, 2, 3, 2, 1, 2, 2, 1, 3, 3, 2,
                                                          2, 2, 3, 1, 1, 1, 2, 2, 3, 3, 3, 3, 1, 3, 2, 1, 3, 1, 3, 1,
                                                          2, 2, 3, 3, 1, 3, 1, 3, 2, 3, 1, 3, 2, 3, 1, 1, 2, 3, 3, 1,
                                                          2, 1, 3, 1, 2, 2, 2, 1, 3, 2, 2, 3, 1, 1, 1, 2, 1, 3, 1, 1])
            self.stakeholders_pbis_priorities = np.vstack((self.stakeholders_pbis_priorities,
                                                           [3, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 3, 3, 2, 1, 3, 2, 3, 3, 1,
                                                            3, 3, 3, 2, 3, 1, 2, 2, 3, 3, 1, 3, 2, 2, 1, 2, 3, 2, 3, 3,
                                                            3, 3, 1, 1, 3, 2, 2, 2, 1, 3, 3, 3, 1, 2, 2, 3, 3, 2, 1, 1,
                                                            1, 3, 2, 3, 1, 2, 1, 2, 3, 1, 1, 3, 1, 3, 2, 1, 3, 3, 1, 2,
                                                            1, 2, 1, 2, 2, 1, 3, 2, 2, 2, 3, 2, 2, 3, 2, 2, 1, 3, 1,
                                                            1]))
            self.stakeholders_pbis_priorities = np.vstack((self.stakeholders_pbis_priorities,
                                                           [1, 1, 1, 2, 1, 1, 1, 3, 2, 2, 3, 3, 3, 1, 3, 1, 2, 2, 3, 3,
                                                            2, 1, 2, 3, 2, 3, 3, 1, 3, 3, 3, 2, 1, 2, 2, 1, 1, 3, 1, 2,
                                                            1, 3, 1, 3, 3, 3, 3, 1, 3, 2, 3, 1, 2, 3, 2, 3, 2, 1, 2, 3,
                                                            1, 1, 2, 3, 3, 1, 3, 3, 3, 1, 3, 1, 3, 1, 1, 2, 3, 3, 1, 2,
                                                            1, 2, 3, 2, 3, 1, 2, 2, 3, 3, 3, 3, 2, 1, 1, 2, 3, 3, 2,
                                                            3]))
            self.stakeholders_pbis_priorities = np.vstack((self.stakeholders_pbis_priorities,
                                                           [3, 2, 2, 1, 3, 1, 3, 2, 3, 2, 3, 2, 1, 3, 2, 3, 2, 1, 3, 3,
                                                            1, 1, 1, 2, 3, 3, 2, 1, 1, 1, 1, 2, 2, 2, 3, 2, 2, 3, 1, 1,
                                                            3, 1, 1, 3, 1, 2, 1, 1, 3, 2, 2, 1, 3, 2, 1, 3, 3, 1, 2, 3,
                                                            2, 2, 3, 3, 3, 1, 2, 1, 2, 1, 2, 3, 3, 2, 2, 2, 1, 3, 3, 1,
                                                            3, 1, 2, 2, 2, 1, 1, 1, 3, 1, 1, 3, 3, 1, 2, 1, 2, 3, 1,
                                                            3]))
            self.stakeholders_pbis_priorities = np.vstack((self.stakeholders_pbis_priorities,
                                                           [1, 2, 3, 1, 3, 1, 2, 3, 1, 1, 2, 2, 3, 1, 2, 1, 1, 1, 1, 3,
                                                            1, 1, 3, 3, 3, 2, 2, 3, 2, 3, 1, 1, 3, 3, 2, 2, 1, 1, 2, 1,
                                                            3, 1, 1, 2, 1, 2, 3, 3, 2, 2, 1, 3, 3, 2, 3, 1, 2, 1, 3, 2,
                                                            2, 2, 1, 2, 1, 3, 2, 1, 2, 1, 2, 2, 3, 2, 1, 3, 2, 3, 1, 3,
                                                            3, 2, 1, 2, 2, 2, 2, 1, 3, 3, 3, 1, 1, 3, 1, 3, 3, 3, 3,
                                                            3]))
        else:
            raise Exception("Sorry, dataset with id=", id, " not found.")

        self.num_pbis = len(self.pbis_cost)
        self.pbis_satisfaction = self.stakeholders_importances.dot(self.stakeholders_pbis_priorities)

        # now two escalation follows, based on
        # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
        # scale pbis cost in range [0-1]
        margin = 1 / self.num_pbis # used to avoid zeros
        diff = np.max(self.pbis_cost) - np.min(self.pbis_cost)
        self.pbis_cost_scaled = (self.pbis_cost - np.min(self.pbis_cost) + margin) / (diff + margin)

        # scale pbis satisfaction in range[0-1]
        diff = np.max(self.pbis_satisfaction) - np.min(self.pbis_satisfaction)
        self.pbis_satisfaction_scaled = (self.pbis_satisfaction - np.min(self.pbis_satisfaction) + margin) / (diff + margin)

        # each pbi score is computed from the scaled versions of pbi satisfaction and cost
        self.pbis_score = self.pbis_satisfaction_scaled / self.pbis_cost_scaled
