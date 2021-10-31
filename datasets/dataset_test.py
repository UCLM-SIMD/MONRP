from typing import Tuple
import numpy as np


def dataset_test() -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    pbis_cost = np.array(
        [3, 3, 3, 3, 3])
    num_pbis = len(pbis_cost)
    stakeholders_importances = np.array([1, 3])
    stakeholders_pbis_priorities = np.array(
        [4, 2, 1, 2, 5])
    stakeholders_pbis_priorities = np.vstack((stakeholders_pbis_priorities,
                                              [4, 2, 1, 2, 5]))
    dependencies = np.array(
        [[2], None, None, [1], None], dtype=object)

    return pbis_cost, num_pbis, stakeholders_importances, stakeholders_pbis_priorities, dependencies
