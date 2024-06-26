import os
import os.path as osp
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
    return np.stack(
        (shift(points, displacement), np.array([color] * len(points))), axis=1
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
