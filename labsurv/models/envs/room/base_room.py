import os
import os.path as osp
from typing import List

import numpy as np
import pandas as pd
from labsurv.utils.surveillance import build_block, shift
from pyntcloud import PyntCloud


class BaseRoom:
    def __init__(self, shape: List[int]):
        if not (
            len(shape) == 3
            and isinstance(shape[0], int)
            and isinstance(shape[1], int)
            and isinstance(shape[2], int)
        ):
            raise ValueError(
                "A room should be in 3d shape with all side lengths integers."
            )

        self.shape = np.array(shape)
        self.occupancy = []
        self.occ_color = []

    def _check_inside_room(self, points: np.ndarray | List[np.ndarray]):
        for point in points:
            assert point.shape == (3,), f"Invalid point coordinate {point}."

            if (
                point[0] < 0
                or point[0] >= self.shape[0]
                or point[1] < 0
                or point[1] >= self.shape[1]
                or point[2] < 0
                or point[2] >= self.shape[2]
            ):
                raise ValueError("Point outside of the room.")

        return True

    def add_block(
        self,
        shape: List[int],
        color: np.ndarray | str | List[int] = np.array([255, 255, 255]),
        near_origin_vertex: np.ndarray | List[int] = np.array([0, 0, 0]),
    ):
        points, point_colors = build_block(shape, color)
        self._check_inside_room(points)

        # consider `near_origin_vertex` as the displacement vector
        points = shift(points, near_origin_vertex)

        for point, point_color in zip(points, point_colors):
            new_point = True
            for index, occ in enumerate(self.occupancy):
                if np.array_equal(point, occ):
                    self.occ_color[index] = point_color
                    new_point = False
                    break

            if new_point:
                self.occupancy.append(point)
                self.occ_color.append(point_color)

    def save(self, save_path: str):
        show_points_with_color = np.stack(
            (self.occupancy, self.occ_color), axis=1
        ).reshape(-1, 6)
        # import pdb; pdb.set_trace()
        show_points_with_color = pd.DataFrame(show_points_with_color)
        show_points_with_color.columns = ["x", "y", "z", "red", "green", "blue"]
        pointcloud = PyntCloud(show_points_with_color)

        pointcloud.add_scalar_field("rgb_intensity")

        if not save_path.endswith(".ply"):
            os.makedirs(save_path, exist_ok=True)
            save_path = osp.join(save_path, "BaseRoom.ply")
        else:
            os.makedirs(osp.dirname(save_path), exist_ok=True)

        pointcloud.to_file(save_path)
