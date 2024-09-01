from typing import Optional

import numpy as np
import pandas as pd
import torch
from labsurv.utils.string import to_filename
from labsurv.utils.surveillance import shift
from pyntcloud import PyntCloud
from torch import Tensor


def concat_points_with_color(
    points: Tensor, color: np.ndarray, displacement: Optional[Tensor] = None
) -> Optional[Tensor]:
    """
    ## Arguments:

        points (Tensor): [W, D, H, ...], torch.int64 or torch.float16. A mask of points
        to be visualized.

        color (np.ndarray): 0-255 RGB representation of the color.

        displacement (Optional[Tensor]): an epsilon to avoid points from being covered
        by others.

    ## Returns:

        points_with_color (Tensor): [N, 6], [x, y, z, R, G, B], torch.float16.
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
    ## Description:

        Save input points_with_color to `ply` file.

    ## Arguments:

        points_with_color (Tensor): [N, 6], [x, y, z, R, G, B], torch.float16.

        save_path (str): path to save the ".ply" file.

        default_filename (str)
    """
    # NOTE(eric): VSCode pointcloud visualization cannot render `torch.float16` points,
    # and the visualization will be fully green screen. Changing to `torch.float` is
    # necessary.
    points_with_color = points_with_color.cpu().type(torch.float)

    framed_points_with_color = pd.DataFrame(points_with_color)
    framed_points_with_color.columns = ["x", "y", "z", "red", "green", "blue"]
    pointcloud = PyntCloud(framed_points_with_color)

    pointcloud.add_scalar_field("rgb_intensity")

    save_ply_path = to_filename(save_path, ".ply", default_filename)

    pointcloud.to_file(save_ply_path)
