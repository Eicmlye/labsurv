from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from labsurv.utils.string import to_filename
from labsurv.utils.surveillance import shift
from matplotlib.colors import LinearSegmentedColormap
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


def apply_colormap_to_3dtensor(
    distribution: Tensor,
    colormap: Optional[LinearSegmentedColormap] = None,
) -> Tensor:
    """
    ## Description:

        Only plot nonzero prob positions.

    ## Arguments:

        distribution (Tensor): [W, D, H], torch.float.

        colormap (Optional[LinearSegmentedColormap])
    """
    if colormap is None:
        colors = [
            "#ff0000",  # red
            "#00ff00",  # green
            "#0000ff",  # blue
        ]
        colormap = LinearSegmentedColormap.from_list("custom", colors, N=256)

    assert isinstance(colormap, LinearSegmentedColormap)

    max_prob: float = distribution.max().item()
    assert max_prob <= 1.0
    distribution[distribution == 0] = 2
    min_prob: float = distribution.min().item()
    distribution[distribution == 2] = 0
    assert min_prob >= 0.0

    # 0 prob positions get negative values
    color_normalized_dist = (distribution - min_prob) / (max_prob - min_prob)

    color_list: List[float] = (
        color_normalized_dist[color_normalized_dist >= 0].view(-1).tolist()
    )
    # now 0 prob positions are ignored
    color_mapping = partial(colormap, bytes=True)
    color_list = list(map(color_mapping, color_list))
    color_list = list(map(lambda x: x[:3], color_list))
    colored_dist = torch.tensor(  # [N, 3]
        color_list, dtype=torch.float, device=distribution.device
    )
    coords = distribution.nonzero()  # [N, 3]
    points_with_color = torch.cat((coords, colored_dist), dim=1)

    return points_with_color  # [W * D * H, 6]


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


def visualize_distribution_heatmap(
    distribution: Tensor,
    save_path: str,
    default_filename: str = "distribution_heatmap",
    colormap: Optional[LinearSegmentedColormap] = None,
):
    """
    ## Arguments:

        distribution (Tensor): [W, D, H], torch.float.

        save_path (str): path to save the ".ply" file.

        default_filename (str)

        colormap (Optional[LinearSegmentedColormap])
    """
    points_with_color = apply_colormap_to_3dtensor(distribution, colormap)
    save_visualized_points(points_with_color, save_path, default_filename)
