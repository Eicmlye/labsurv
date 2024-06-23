import os
import os.path as osp
import pickle
from typing import List

import numpy as np
import pandas as pd
from labsurv.utils.surveillance import build_block, shift
from labsurv.utils.surveillance.build import COLOR_MAP
from pyntcloud import PyntCloud


class SurveillanceRoom:
    POINT_TYPE = dict(
        occupancy="grey",
        install_permitted="yellow",
        must_monitor="red",
    )

    def __init__(self, shape: List[int] | None = None, load_from: str | None = None):
        """
        Argument:

            shape (List[int] | None): If is not `None`, a `Room` of `shape` size will
            be generated.

            load_from (str | None): only valid if `shape` is `None`, and the `Room`
            will be loaded from `load_from`.

        Attributes:

            shape (np.ndarray): the shape of the room.

            occupancy (List[np.ndarray]): [array([x, y, z]), ...], the coordinates of
            the points occupied by objects.

            install_permitted (List[np.ndarray]): [array([x, y, z]), ...], the
            coordinates of the points that allow cameras to be installed at.

            must_monitor (List[np.ndarray]): [array([x, y, z]), ...], the coordinates
            of the points that must be monitored by at least 1 camera.
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
            self.install_permitted = []
            self.must_monitor = []
        else:
            if not osp.exists(load_from):
                raise ValueError(f"The path {load_from} does not exist.")
            if not load_from.endswith(".pkl"):
                raise ValueError("Only `pkl` files can be loaded.")

            with open(load_from, "rb") as fpkl:
                room_data = pickle.load(fpkl)
                self.shape = room_data["shape"]
                self.occupancy = room_data["occupancy"]
                self.install_permitted = room_data["install_permitted"]
                self.must_monitor = room_data["must_monitor"]

    @property
    def occupancy_color(self):
        return COLOR_MAP[self.POINT_TYPE["occupancy"]]

    @property
    def install_permitted_color(self):
        return COLOR_MAP[self.POINT_TYPE["install_permitted"]]

    @property
    def must_monitor_color(self):
        return COLOR_MAP[self.POINT_TYPE["must_monitor"]]

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
        point_type: str = "occupancy",
        near_origin_vertex: np.ndarray | List[int] = np.array([0, 0, 0]),
    ):
        if point_type not in self.POINT_TYPE.keys():
            raise ValueError("SurveillanceRoom does not allow customized color points.")

        points, _ = build_block(shape, self.POINT_TYPE[point_type])
        self._check_inside_room(points)

        # consider `near_origin_vertex` as the displacement vector
        points = shift(points, near_origin_vertex)

        # list object is mutable, so `target_point_list` is a pointer
        target_point_list = None
        if point_type == "occupancy":
            target_point_list = self.occupancy
        elif point_type == "install_permitted":
            target_point_list = self.install_permitted
        elif point_type == "must_monitor":
            target_point_list = self.must_monitor

        for point in points:
            new_point = True
            for target_point in target_point_list:
                if np.array_equal(point, target_point):
                    new_point = False
                    break

            if new_point:
                target_point_list.append(point)

    def save(self, save_path: str):
        """
        Saves the `Room` class data in a `pkl` file.
        """
        if not save_path.endswith(".pkl"):
            os.makedirs(save_path, exist_ok=True)
            save_pkl_path = osp.join(save_path, "SurveillanceRoom.pkl")
        else:
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            save_pkl_path = save_path

        with open(save_pkl_path, "wb") as fpkl:
            pickle.dump(
                dict(
                    shape=self.shape,
                    occupancy=self.occupancy,
                    install_permitted=self.install_permitted,
                    must_monitor=self.must_monitor,
                ),
                fpkl,
            )

    def visualize(self, save_path: str):
        """
        Saves the pointcloud in a `ply` file.
        """
        EPSILON = np.array([0.1, 0, 0])
        occupancy_with_color = np.stack(
            (self.occupancy, np.array([self.occupancy_color] * len(self.occupancy))),
            axis=1,
        ).reshape(-1, 6)
        install_permitted_with_color = np.stack(
            (
                shift(self.install_permitted, EPSILON),
                np.array([self.install_permitted_color] * len(self.install_permitted)),
            ),
            axis=1,
        ).reshape(-1, 6)
        must_monitor_with_color = np.stack(
            (
                shift(self.must_monitor, -EPSILON),
                np.array([self.must_monitor_color] * len(self.must_monitor)),
            ),
            axis=1,
        ).reshape(-1, 6)
        # import pdb; pdb.set_trace()
        show_points_with_color = np.row_stack(
            (
                occupancy_with_color,
                install_permitted_with_color,
                must_monitor_with_color,
            ),
        )
        show_points_with_color = pd.DataFrame(show_points_with_color)
        show_points_with_color.columns = ["x", "y", "z", "red", "green", "blue"]
        pointcloud = PyntCloud(show_points_with_color)

        pointcloud.add_scalar_field("rgb_intensity")

        if not save_path.endswith(".ply"):
            os.makedirs(save_path, exist_ok=True)
            save_ply_path = osp.join(save_path, "SurveillanceRoom.ply")
        else:
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            save_ply_path = save_path

        pointcloud.to_file(save_ply_path)
