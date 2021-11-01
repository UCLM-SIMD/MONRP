from typing import Tuple
import numpy as np


def dataset_p2() -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    pbis_cost = np.array([16, 19, 16, 7, 19, 15, 8, 10, 6, 18, 15, 12, 16, 20, 9, 4, 16, 2, 9, 3,
                          2, 10, 4, 2, 7, 15, 8, 20, 9, 11, 5, 1, 17, 6, 2, 16, 8, 12, 18, 5, 6,
                          14, 15, 20, 14, 9, 16, 6, 6, 6, 6, 2, 17, 8, 1, 3, 14, 16, 18, 7, 10, 7,
                          16, 19, 17, 15, 11, 8, 20, 1, 5, 8, 3, 15, 4, 20, 10, 20, 3, 20, 10, 16,
                          19, 3, 12, 16, 15, 1, 6, 7, 15, 18, 4, 7, 2, 7, 8, 7, 7, 3])
    num_pbis = len(pbis_cost)
    stakeholders_importances = np.array([1, 5, 3, 3, 1])

    stakeholders_pbis_priorities = np.array([1, 2, 1, 1, 2, 3, 3, 1, 1, 3, 1, 1, 3, 2, 3, 2, 2, 3, 1, 3, 2, 1, 1, 1, 3, 3, 3, 3, 1, 2, 2, 3, 2, 1, 2, 2, 1, 3, 3, 2, 2, 2, 3,
                                             1, 1, 1, 2, 2, 3, 3, 3, 3, 1, 3, 2, 1, 3, 1, 3, 1, 2, 2, 3, 3, 1, 3, 1, 3, 2, 3, 1, 3, 2, 3, 1, 1, 2, 3, 3, 1, 2, 1, 3, 1, 2, 2, 2, 1, 3, 2, 2, 3, 1, 1, 1, 2, 1, 3, 1, 1])
    stakeholders_pbis_priorities = np.vstack((stakeholders_pbis_priorities,
                                              [3, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 3, 3, 2, 1, 3, 2, 3, 3, 1, 3, 3, 3, 2, 3, 1, 2, 2, 3, 3, 1, 3, 2, 2, 1, 2, 3, 2, 3, 3, 3, 3, 1, 1, 3, 2, 2, 2, 1, 3, 3, 3, 1, 2, 2, 3, 3, 2, 1, 1, 1, 3, 2, 3, 1, 2, 1, 2, 3, 1, 1, 3, 1, 3, 2, 1, 3, 3, 1, 2, 1, 2, 1, 2, 2, 1, 3, 2, 2, 2, 3, 2, 2, 3, 2, 2, 1, 3, 1, 1]))
    stakeholders_pbis_priorities = np.vstack((stakeholders_pbis_priorities,
                                              [1, 1, 1, 2, 1, 1, 1, 3, 2, 2, 3, 3, 3, 1, 3, 1, 2, 2, 3, 3, 2, 1, 2, 3, 2, 3, 3, 1, 3, 3, 3, 2, 1, 2, 2, 1, 1, 3, 1, 2, 1, 3, 1, 3, 3, 3, 3, 1, 3, 2, 3, 1, 2, 3, 2, 3, 2, 1, 2, 3, 1, 1, 2, 3, 3, 1, 3, 3, 3, 1, 3, 1, 3, 1, 1, 2, 3, 3, 1, 2, 1, 2, 3, 2, 3, 1, 2, 2, 3, 3, 3, 3, 2, 1, 1, 2, 3, 3, 2, 3]))
    stakeholders_pbis_priorities = np.vstack((stakeholders_pbis_priorities,
                                              [3, 2, 2, 1, 3, 1, 3, 2, 3, 2, 3, 2, 1, 3, 2, 3, 2, 1, 3, 3, 1, 1, 1, 2, 3, 3, 2, 1, 1, 1, 1, 2, 2, 2, 3, 2, 2, 3, 1, 1, 3, 1, 1, 3, 1, 2, 1, 1, 3, 2, 2, 1, 3, 2, 1, 3, 3, 1, 2, 3, 2, 2, 3, 3, 3, 1, 2, 1, 2, 1, 2, 3, 3, 2, 2, 2, 1, 3, 3, 1, 3, 1, 2, 2, 2, 1, 1, 1, 3, 1, 1, 3, 3, 1, 2, 1, 2, 3, 1, 3]))
    stakeholders_pbis_priorities = np.vstack((stakeholders_pbis_priorities,
                                              [1, 2, 3, 1, 3, 1, 2, 3, 1, 1, 2, 2, 3, 1, 2, 1, 1, 1, 1, 3, 1, 1, 3, 3, 3, 2, 2, 3, 2, 3, 1, 1, 3, 3, 2, 2, 1, 1, 2, 1, 3, 1, 1, 2, 1, 2, 3, 3, 2, 2, 1, 3, 3, 2, 3, 1, 2, 1, 3, 2, 2, 2, 1, 2, 1, 3, 2, 1, 2, 1, 2, 2, 3, 2, 1, 3, 2, 3, 1, 3, 3, 2, 1, 2, 2, 2, 2, 1, 3, 3, 3, 1, 1, 3, 1, 3, 3, 3, 3, 3]))

    dependencies = np.array([None, [24], [26, 27, 28, 29], [5], None, [7], [30], None, None, [32, 33],  # 1-10
                             None, None, None, [32, 34, 37, 38], None, [
        39, 40], [43], None, None, None,  # 11-20
        [22], [21], None, None, None, None, None, None, [
        49, 50, 51], [52, 53],  # 21-30
        [55], [56, 57, 33], [58, 32], None, None, [
        61], None, None, [63], [64],
        None, None, [65], None, None, [
        68, 47], [70, 46], None, None, None,
        None, None, None, None, [79], [
        80], [80], None, None, None,
        None, [83, 84], None, [87], [66], [
        65], None, None, None, None,
        None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None,
    ], dtype=object)

    return pbis_cost, num_pbis, stakeholders_importances, stakeholders_pbis_priorities, dependencies
