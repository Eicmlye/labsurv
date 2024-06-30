import os
import os.path as osp
import pickle
from typing import List

import numpy as np
import pandas as pd
from labsurv.utils.surveillance import build_block, shift
from pyntcloud import PyntCloud


class BaseRoom:
    def __init__(self, shape: List[int] | None = None, load_from: str | None = None):
        """
        Argument:

            shape (List[int] | None): If is not `None`, a `Room` of `shape` size will
            be generated.

            load_from (str | None): only valid if `shape` is `None`, and the `Room`
            will be loaded from `load_from`.

        Attributes:

            shape (np.ndarray): the shape of the room.

            occupancy (List[np.ndarray]): [array([x, y, z]), ...], the coordinates of the points occupied by objects.

            occ_color (List[np.ndarray]): [array([r, g, b]), ...], the color of every point in `self.occupancy`.
        """

        assert (
            shape is not None or load_from is not None
        ), "At least one of `shape` and `load_from` should be specified."

        if shape is not None:
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
        else:
            if not osp.exists(load_from):
                raise ValueError(f"The path {load_from} does not exist.")
            if not load_from.endswith(".pkl"):
                raise ValueError("Only `pkl` files can be loaded.")

            with open(load_from, "rb") as fpkl:
                room_data = pickle.load(fpkl)
                self.shape = room_data["shape"]
                self.occupancy = room_data["occupancy"]
                self.occ_color = room_data["occ_color"]

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
        color: np.ndarray | str | List[int] = np.array([128, 128, 128]),
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
        """
        Saves the `Room` class data in a `pkl` file.
        """
        if not save_path.endswith(".pkl"):
            os.makedirs(save_path, exist_ok=True)
            save_pkl_path = osp.join(save_path, "BaseRoom.pkl")
        else:
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            save_pkl_path = save_path

        with open(save_pkl_path, "wb") as fpkl:
            pickle.dump(
                dict(
                    shape=self.shape,
                    occupancy=self.occupancy,
                    occ_color=self.occ_color,
                ),
                fpkl,
            )

    def visualize(self, save_path: str):
        """
        Saves the pointcloud in a `ply` file.
        """
        show_points_with_color = np.stack(
            (self.occupancy, self.occ_color), axis=1
        ).reshape(-1, 6)
        show_points_with_color = pd.DataFrame(show_points_with_color)
        show_points_with_color.columns = ["x", "y", "z", "red", "green", "blue"]
        pointcloud = PyntCloud(show_points_with_color)

        pointcloud.add_scalar_field("rgb_intensity")

        if not save_path.endswith(".ply"):
            os.makedirs(save_path, exist_ok=True)
            save_ply_path = osp.join(save_path, "BaseRoom.ply")
        else:
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            save_ply_path = save_path

        pointcloud.to_file(save_ply_path)
