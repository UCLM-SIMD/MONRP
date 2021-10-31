from typing import Tuple
import numpy as np


def dataset_p1() -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    pbis_cost = np.array(
        [1, 4, 2, 3, 4, 7, 10, 2, 1, 3, 2, 5, 8, 2, 1, 4, 10, 4, 8, 4])
    num_pbis = len(pbis_cost)
    stakeholders_importances = np.array([1, 4, 2, 3, 4])
    stakeholders_pbis_priorities = np.array(
        [4, 2, 1, 2, 5, 5, 2, 4, 4, 4, 2, 3, 4, 2, 4, 4, 4, 1, 3, 2])
    stakeholders_pbis_priorities = np.vstack((stakeholders_pbis_priorities,
                                              [4, 4, 2, 2, 4, 5, 1, 4, 4, 5, 2, 3, 2, 4, 4, 2, 3, 2, 3,
                                               1]))
    stakeholders_pbis_priorities = np.vstack((stakeholders_pbis_priorities,
                                              [5, 3, 3, 3, 4, 5, 2, 4, 4, 4, 2, 4, 1, 5, 4, 1, 2, 3, 3,
                                               2]))
    stakeholders_pbis_priorities = np.vstack((stakeholders_pbis_priorities,
                                              [4, 5, 2, 3, 3, 4, 2, 4, 2, 3, 5, 2, 3, 2, 4, 3, 5, 4, 3,
                                               2]))
    stakeholders_pbis_priorities = np.vstack((stakeholders_pbis_priorities,
                                              [5, 4, 2, 4, 5, 4, 2, 4, 5, 2, 4, 5, 3, 4, 4, 1, 1, 2, 4,
                                               1]))
    # dependencies = np.array(
    #    [None,None,[["combination",12]],[["implication",8],["implication",17]],None,None,None,[["implication",17]],
    #    [["implication",3],["implication",6],["implication",12]
    #    ,["implication",19]],None,[["implication",19],["combination",13]],None,None,None,None,None,None,None,None,None],dtype=object)
    dependencies = np.array([None, None, [12], [8, 17], None, None, None, [17], [3, 6, 12, 19], None,
                             [19, 13], [3], [11], None, None, None, None, None, None, None, ], dtype=object)

    return pbis_cost, num_pbis, stakeholders_importances, stakeholders_pbis_priorities, dependencies
