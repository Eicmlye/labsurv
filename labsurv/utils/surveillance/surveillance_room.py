import json
import os
import os.path as osp
import pickle
from typing import List

import numpy as np
from labsurv.utils.surveillance import (
    COLOR_MAP,
    build_block,
    concat_points_with_color,
    save_visualized_points,
    shift,
)


class SurveillanceRoom:
    _POINT_TYPE = dict(
        occupancy="grey",
        install_permitted="yellow",
        must_monitor="red",
        camera="orange",
        visible="green",
    )

    def __init__(
        self,
        cam_intrinsics_path: str,
        shape: List[int] | None = None,
        load_from: str | None = None,
    ):
        """
        Description:

            The `SurveillanceRoom` class is responsible for the geometric data and
            camera settings of the room interior. It is not necessary to be a SINGLE
            room. One may make it a large region of interest and add blocks to separate
            different rooms.

        Argument:

            cam_intrinsics_path (str): the configuration file path of the clip size,
            focal length and other parameters of all the types of cameras used in this
            room.

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

            must_monitor (List[np.ndarray]):
            [array([x, y, z, h_res_req, v_res_req]), ...], the coordinates of the
            points that must be monitored by at least 1 camera, along with the
            horizontal and vertical resolution requirements at this position.

            cam_extrinsics (List[np.ndarray]): [array([x, y, z, pan, tilt]), ...], the
            position and orientation of the cameras installed.

            cam_types (List[str]): the types of cameras installed.
        """

        if shape is None and load_from is None:
            raise ValueError(
                "At least one of `shape` and `load_from` should be specified."
            )

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

            self.cam_extrinsics = []
            self.cam_types = []
            self.visible_points = []
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

            self.cam_extrinsics = room_data["cam_extrinsics"]
            self.cam_types = room_data["cam_types"]
            self.visible_points = room_data["visible_points"]

        with open(cam_intrinsics_path, "r") as f:
            self._CAM_INTRINSICS = json.load(f)
        if not isinstance(self._CAM_INTRINSICS, dict):
            raise ValueError(
                "Loaded cam intrinsics should be a dict, "
                f"not {type(self._CAM_INTRINSICS)}."
            )

    @property
    def occupancy_color(self):
        return COLOR_MAP[self._POINT_TYPE["occupancy"]]

    @property
    def install_permitted_color(self):
        return COLOR_MAP[self._POINT_TYPE["install_permitted"]]

    @property
    def must_monitor_color(self):
        return COLOR_MAP[self._POINT_TYPE["must_monitor"]]

    @property
    def camera_color(self):
        return COLOR_MAP[self._POINT_TYPE["camera"]]

    @property
    def visible_color(self):
        return COLOR_MAP[self._POINT_TYPE["visible"]]

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
                return False

        return True

    def _check_permit_install(self, cam_pos: np.ndarray):
        for permit_pos in self.install_permitted:
            if np.array_equal(cam_pos, permit_pos):
                return True

        return False

    def _compute_visibility(self):
        pass

    def add_block(
        self,
        shape: List[int],
        point_type: str = "occupancy",
        near_origin_vertex: np.ndarray | List[int] = np.array([0, 0, 0]),
        **kwargs,
    ):
        if point_type not in self._POINT_TYPE.keys():
            raise ValueError("SurveillanceRoom does not allow customized color points.")

        points, _ = build_block(shape, self._POINT_TYPE[point_type])
        if not self._check_inside_room(points):
            raise ValueError("Point outside of the room.")

        # consider `near_origin_vertex` as the displacement vector
        points = shift(points, near_origin_vertex)

        # WARN: list object is mutable, so `target_point_list` is a pointer
        target_point_list = None
        if point_type == "occupancy":
            target_point_list = self.occupancy
        elif point_type == "install_permitted":
            target_point_list = self.install_permitted
        elif point_type == "must_monitor":
            target_point_list = self.must_monitor

        if len(kwargs.keys()) > 0:
            appendix = np.array([kwargs[key] for key in kwargs.keys()])

        for point in points:
            new_point = True
            for target_point in target_point_list:
                if np.array_equal(point, target_point):
                    new_point = False
                    break

            if new_point:
                if len(kwargs.keys()) > 0:
                    target_point_list.append(np.hstack((point, appendix)))
                else:
                    target_point_list.append(point)

    def add_cam(
        self,
        pos: List[int] | np.ndarray,
        direction: List[float] | np.ndarray,
        cam_type: str,
    ):
        """
        Arguments:

            pos (List[int] | np.ndarray): the position of the camera. Must be on the
            room points.

            direction (List[float] | np.ndarray): the pan and tilt angle of the camera.

            cam_type (str): the type of camera.
        """
        if cam_type not in self._CAM_INTRINSICS.keys():
            raise ValueError(f"SurveillanceRoom does not support {cam_type} cameras.")

        if isinstance(pos, List):
            pos = np.array(pos)
        if not self._check_inside_room([pos]):
            raise ValueError("Point outside of the room.")
        if not self._check_permit_install(pos):
            raise ValueError(f"{pos} is not permitted to install.")

        if isinstance(direction, List):
            direction = np.array(direction)
        extrinsics = np.hstack((pos, direction))

        for cam_extrinsic in self.cam_extrinsics:
            if np.array_equal(extrinsics, cam_extrinsic):
                raise ValueError(f"Existed camera found at {pos}.")

        self.cam_extrinsics.append(extrinsics)
        self.cam_types.append(cam_type)

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
                    cam_extrinsics=self.cam_extrinsics,
                    cam_types=self.cam_types,
                    visible_points=self.visible_points,
                ),
                fpkl,
            )

    def visualize(self, save_path: str, mode: str = "occupancy"):
        """
        Saves the pointcloud in a `ply` file.
        """

        if mode == "occupancy":
            points_with_color = self.visualize_occupancy()
        elif mode == "camera":
            points_with_color = self.visualize_camera()
        else:
            raise ValueError(f"Unsupported mode {mode}.")

        save_visualized_points(points_with_color, save_path)

    def visualize_occupancy(self):
        """
        Visualize occupancy, install_permitted and must_monitor.
        """

        EPSILON = np.array([0.1, 0, 0])
        occupancy_with_color = concat_points_with_color(
            self.occupancy, self.occupancy_color
        )
        install_permitted_with_color = concat_points_with_color(
            self.install_permitted, self.install_permitted_color, EPSILON
        )
        must_monitor_with_color = concat_points_with_color(
            [pos[:3] for pos in self.must_monitor], self.must_monitor_color, -EPSILON
        )

        points_with_color = np.row_stack(
            (
                occupancy_with_color,
                install_permitted_with_color,
                must_monitor_with_color,
            ),
        )

        return points_with_color

    def visualize_camera(self):
        """
        Visualize occupancy, camera pos and visible points.
        """

        EPSILON = np.array([0.1, 0, 0])
        occupancy_with_color = concat_points_with_color(
            self.occupancy, self.occupancy_color
        )
        cameras_with_color = concat_points_with_color(
            [extrinsics[:3] for extrinsics in self.cam_extrinsics],
            self.camera_color,
            EPSILON,
        )
        visible_with_color = concat_points_with_color(
            self.visible_points, self.visible_color, -EPSILON
        )

        points_with_color = np.row_stack(
            (
                occupancy_with_color,
                cameras_with_color,
                visible_with_color,
            ),
        )

        return points_with_color
