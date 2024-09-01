import os
import os.path as osp
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import torch
from labsurv.utils.surveillance import shift
from pyntcloud import PyntCloud
from torch import Tensor


def concat_points_with_color(
    points: Tensor, color: np.ndarray, displacement: Optional[Tensor] = None
) -> Optional[Tensor]:
    """
    # Arguments:

        `points` (Tensor): Width * Depth * Height tensor
    """
    device = points.device
    if displacement is None:
        displacement = torch.tensor([0, 0, 0], dtype=torch.float16, device=device)
    elif displacement.device != device:
        displacement = displacement.to(device)
    CENTER_SHIFT = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float16, device=device)

    if points.ndim > 3:
        points = points[:, :, :, 0]
    elif points.ndim < 3:
        raise ValueError("No valid 3D coordinates found in given point list.")

    points = points.nonzero()
    if points.numel() == 0:
        return None

    points = shift(points, displacement + CENTER_SHIFT)
    colors = torch.tensor(color, dtype=torch.float16, device=device).repeat(
        len(points), 1
    )

    return torch.cat((points, colors), 1)


def save_visualized_points(
    points_with_color: Tensor, save_path: str, default_filename: str = "pointcloud"
):
    """
    Save input points_with_color to `ply` file.
    """
    # NOTE(eric): VSCode pointcloud visualization cannot render `torch.float16` points,
    # and the visualization will be fully green screen. Changing to `torch.float` is
    # necessary.
    points_with_color = points_with_color.cpu().type(torch.float)

    framed_points_with_color = pd.DataFrame(points_with_color)
    framed_points_with_color.columns = ["x", "y", "z", "red", "green", "blue"]
    pointcloud = PyntCloud(framed_points_with_color)

    pointcloud.add_scalar_field("rgb_intensity")
    # print(pointcloud)

    if not save_path.endswith(".ply"):
        os.makedirs(save_path, exist_ok=True)
        save_ply_path = osp.join(save_path, f"{default_filename}.ply")
    else:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        save_ply_path = save_path

    pointcloud.to_file(save_ply_path)
