from typing import List

import numpy as np


def shift(points: List[np.ndarray], displacement: np.ndarray):
    for point in points:
        point += displacement

    return points


def rotate(points: List[np.ndarray], rot_mat: np.ndarray):
    pass
