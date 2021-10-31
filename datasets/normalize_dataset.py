from typing import Tuple
import numpy as np


def normalize_dataset(pbis_cost: np.ndarray, stakeholders_importances: np.ndarray, stakeholders_pbis_priorities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_pbis = len(pbis_cost)

    pbis_satisfaction = stakeholders_importances.dot(
        stakeholders_pbis_priorities)

    # now two escalation follows, based on
    # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    # scale pbis cost in range [0-1]
    margin = 1 / num_pbis  # used to avoid zeros
    # suma costes -minimos costes TODO
    diff = np.sum(pbis_cost) - np.min(pbis_cost)
    pbis_cost_scaled = (pbis_cost - np.min(pbis_cost) +
                        margin) / (diff + margin)

    # scale pbis satisfaction in range[0-1]
    # suma satisfact - min TODO
    diff = np.sum(pbis_satisfaction) - np.min(pbis_satisfaction)
    pbis_satisfaction_scaled = (
        pbis_satisfaction - np.min(pbis_satisfaction) + margin) / (diff + margin)

    # each pbi score is computed from the scaled versions of pbi satisfaction and cost
    pbis_score = pbis_satisfaction_scaled / pbis_cost_scaled

    return pbis_cost_scaled, pbis_satisfaction_scaled, pbis_score
