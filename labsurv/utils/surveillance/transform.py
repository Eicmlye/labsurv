from copy import deepcopy
from typing import List

import numpy as np


def shift(points: List[np.ndarray], displacement: np.ndarray):
    result_points = deepcopy(points)
    if np.array_equal(displacement, np.array([0, 0, 0])):
        return result_points

    for index, point in enumerate(result_points):
        result_points[index] = point + displacement

    return result_points


def rotate(points: List[np.ndarray], rot_mat: np.ndarray):
    pass
