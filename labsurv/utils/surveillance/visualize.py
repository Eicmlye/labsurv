import os
import os.path as osp
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
from labsurv.utils.surveillance import shift
from pyntcloud import PyntCloud


def concat_points_with_color(
    points: np.ndarray | List[np.ndarray],
    color: np.ndarray,
    displacement: np.ndarray = np.array([0, 0, 0]),
):
    if len(points) == 0:
        return None

    CENTER_SHIFT = np.array([0.5, 0.5, 0.5])
    result_points = deepcopy(points)
    if len(result_points[0]) > 3:
        result_points = [result_point[:3] for result_point in result_points]
    elif len(result_points[0]) < 3:
        raise ValueError("No valid 3D coordinates found in given point list.")

    return np.stack(
        (
            shift(result_points, displacement + CENTER_SHIFT),
            np.array([color] * len(result_points)),
        ),
        axis=1,
    ).reshape(-1, 6)


def save_visualized_points(
    points_with_color: np.ndarray, save_path: str, default_filename: str = "pointcloud"
):
    """
    Save input points_with_color to `ply` file.
    """
    framed_points_with_color = pd.DataFrame(points_with_color)
    framed_points_with_color.columns = ["x", "y", "z", "red", "green", "blue"]
    pointcloud = PyntCloud(framed_points_with_color)

    pointcloud.add_scalar_field("rgb_intensity")

    if not save_path.endswith(".ply"):
        os.makedirs(save_path, exist_ok=True)
        save_ply_path = osp.join(save_path, f"{default_filename}.ply")
    else:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        save_ply_path = save_path

    pointcloud.to_file(save_ply_path)
