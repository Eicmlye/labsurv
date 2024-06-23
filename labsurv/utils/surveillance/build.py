from typing import List

import numpy as np

COLOR_MAP = dict(
    red=np.array([255, 0, 0]),
    green=np.array([0, 255, 0]),
    blue=np.array([0, 0, 255]),
    yellow=np.array([128, 128, 0]),
    black=np.array([0, 0, 0]),
    white=np.array([255, 255, 255]),
)


def build_block(
    shape: List[int],
    color: np.ndarray | str | List[int] = np.array([0, 0, 0]),
) -> np.ndarray:
    """
    Returns:
        points (np.ndarray): an N * 3 array with all the point coordinates in the block
        given.
    """
    if len(shape) != 3 or not (
        isinstance(shape[0], int)
        and isinstance(shape[1], int)
        and isinstance(shape[2], int)
    ):
        raise ValueError(
            "A block should be in 3d shape with all side lengths integers."
        )

    if isinstance(color, str):
        if color.lower() not in COLOR_MAP.keys():
            raise ValueError(f"Unknown color string {color}, use rgb instead.")
        color = COLOR_MAP[color]
    elif isinstance(color, List):
        color = np.array(color)

    # resolution of the block
    x_res, y_res, z_res = shape

    x_mesh, y_mesh, z_mesh = np.meshgrid(
        np.linspace(0, x_res - 1, x_res),
        np.linspace(0, y_res - 1, y_res),
        np.linspace(0, z_res - 1, z_res),
    )
    points = np.column_stack((x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()))
    colors = np.array([color] * len(points))

    return points, colors
