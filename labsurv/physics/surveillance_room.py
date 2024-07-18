import math
import os
import os.path as osp
import pickle
from typing import List

import numpy as np
from labsurv.utils.string import WARN
from labsurv.utils.surveillance import (
    COLOR_MAP,
    build_block,
    compute_single_cam_visibility,
    concat_points_with_color,
    save_visualized_points,
    shift,
)
from mmcv import Config


class SurveillanceRoom:
    def __init__(
        self,
        cfg_path: str | None = None,
        shape: List[int] | None = None,
        load_from: str | None = None,
    ):
        """
        ## Description:

            The `SurveillanceRoom` class is responsible for the geometric data and
            camera settings of the room interior. It is not necessary to be a SINGLE
            room. One may make it a large region of interest and add blocks to separate
            different rooms.

        ## Argument:

            cfg_path (str): the configuration file path of the clip size, focal length
            and other parameters of the cameras and the room.

            shape (List[int] | None): If is not `None`, a `Room` of `shape` size will
            be generated. Physically, the `Room` is of shape `voxel_length * shape`,
            where the voxel centers are treated as points in the `Room`.

            load_from (str | None): only valid if `shape` is `None`, and the `Room`
            will be loaded from `load_from`.

        ## Attributes:

            shape (np.ndarray): the shape of the room.

            occupancy (List[np.ndarray]): [array([x, y, z]), ...], the coordinates of
            the points occupied by objects.

            install_permitted (List[np.ndarray]): [array([x, y, z]), ...], the
            coordinates of the points that allow cameras to be installed at.

            must_monitor (List[np.ndarray]):
            [array([x, y, z, h_res_req_min/max, v_res_req_min/max]), ...], the
            coordinates of the points that must be monitored by at least 1 camera,
            along with the horizontal and vertical pixel resolution requirements at
            this position.

            cam_extrinsics (List[np.ndarray]): [array([x, y, z, pan, tilt]), ...], the
            position and orientation of the cameras installed.

            cam_types (List[str]): the types of cameras installed.

            visible_points (set): if must_monitor point is visible to camera, its index
            is added to this set.
        """

        if (cfg_path is None or shape is None) and load_from is None:
            raise ValueError(
                "Either (`cfg_path`, `shape`) or `load_from` should be specified."
            )
        if cfg_path is not None and shape is not None and load_from is not None:
            print(
                WARN(
                    "Both (`cfg_path`, `shape`) and `load_from` are specified. "
                    "Only use `load_from`."
                )
            )

        if load_from is None:
            if not (
                len(shape) == 3
                and isinstance(shape[0], int)
                and isinstance(shape[1], int)
                and isinstance(shape[2], int)
            ):
                raise ValueError(
                    "A room should be in 3d shape with all side lengths integers."
                )

            self.cfg_path = cfg_path

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
            self.cfg_path = room_data["cfg_path"]

            self.shape = room_data["shape"]
            self.occupancy = room_data["occupancy"]
            self.install_permitted = room_data["install_permitted"]
            self.must_monitor = room_data["must_monitor"]

            self.cam_extrinsics = room_data["cam_extrinsics"]
            self.cam_types = room_data["cam_types"]
            self.visible_points = room_data["visible_points"]

        cfg = Config.fromfile(self.cfg_path)
        self._CAM_INTRINSICS = cfg.cam_intrinsics
        self._POINT_CONFIGS = cfg.point_configs
        self.voxel_length = cfg.voxel_length
        if not isinstance(self._CAM_INTRINSICS, dict):
            raise ValueError(
                "Loaded cam intrinsics should be a dict, "
                f"not {type(self._CAM_INTRINSICS)}."
            )

    def get_color(self, point_type: str):
        return COLOR_MAP[self._POINT_CONFIGS[point_type]["color"]]

    def get_extra_params_namelist(self, point_type: str):
        return self._POINT_CONFIGS[point_type]["extra_params"]

    @property
    def point_types(self):
        return self._POINT_CONFIGS.keys()

    def _check_point_type(self, point_type: str):
        return point_type in self.point_types

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

    def _check_match_extra_params(
        self, point_type: str, **extra_params
    ) -> int | List[int]:
        provided = set(extra_params.keys())
        if "extra_params" not in self._POINT_CONFIGS[point_type].keys():
            if len(extra_params.keys()) == 0:
                return provided

            return [set(), provided]

        required = set(self._POINT_CONFIGS[point_type]["extra_params"])

        if required == provided:
            return provided

        return [required, provided]

    def _compute_visibility(self):
        vis_dict = dict()
        for index, cam_extrinsic in enumerate(self.cam_extrinsics):
            vis_dict[index] = compute_single_cam_visibility(
                cam_extrinsic,
                self._CAM_INTRINSICS[self.cam_types[index]],
                self.occupancy,
                self.must_monitor,
                self.voxel_length,
            )

        return vis_dict

    def add_block(
        self,
        shape: List[int],
        point_type: str = "occupancy",
        near_origin_vertex: np.ndarray | List[int] = np.array([0, 0, 0]),
        **kwargs,
    ):
        if not self._check_point_type(point_type):
            raise ValueError(
                f"SurveillanceRoom does not support {point_type} type points."
            )

        points, _ = build_block(shape, self.get_color(point_type))
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

        extra_params = self._check_match_extra_params(point_type, **kwargs)
        if isinstance(extra_params, set):
            if len(extra_params) > 0:
                try:
                    appendix = np.array(
                        [
                            kwargs[key]
                            for key in self.get_extra_params_namelist(point_type)
                        ]
                    )
                except KeyError:
                    raise ValueError(
                        f"Extra params key mismatch for {point_type}. "
                        f"Expected {extra_params[0]}, got {extra_params[1]}."
                    )
        else:
            raise ValueError(
                f"Extra params number mismatch for {point_type}. "
                f"Expected {extra_params[0]}, got {extra_params[1]}."
            )

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

        if len(self.cam_extrinsics) > 0:
            vis_dict = self._compute_visibility()
            self.visible_points = set()
            for cam_index, target_dict in vis_dict.items():
                for target_index in target_dict.keys():
                    if target_dict[target_index]:
                        self.visible_points.add(target_index)

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
            pan in (-pi, pi], tilt in [-pi/2, pi/2].

            cam_type (str): the type of camera.
        """
        assert direction[0] > -math.pi and direction[0] <= math.pi
        assert direction[1] >= -math.pi / 2 and direction[1] <= math.pi / 2

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

        vis_dict = self._compute_visibility()
        self.visible_points = set()
        for cam_index, target_dict in vis_dict.items():
            for target_index in target_dict.keys():
                if target_dict[target_index]:
                    self.visible_points.add(target_index)

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
                    cfg_path=self.cfg_path,
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

        ## Arguments:

            save_path (str): if ends with `.ply`, the exact file name will be used,
            otherwise the path is treated as a directory and `SurveillanceRoom.ply` is
            created inside this directory.

            mode (str): specifying what to visualize. Supported mode: `occupancy`,
            `camera`.
        """

        if mode == "occupancy":
            points_with_color = self.visualize_occupancy()
        elif mode == "camera":
            points_with_color = self.visualize_camera()
        else:
            raise ValueError(f"Unsupported mode {mode}.")

        save_visualized_points(
            points_with_color, save_path, "SurveillanceRoom_" + mode[:3]
        )

    def visualize_occupancy(self):
        """
        Visualize occupancy, install_permitted and must_monitor.
        """

        EPSILON = np.array([0.1, 0, 0])
        occupancy_with_color = concat_points_with_color(
            self.occupancy, self.get_color("occupancy")
        )
        install_permitted_with_color = concat_points_with_color(
            self.install_permitted, self.get_color("install_permitted"), EPSILON
        )
        must_monitor_with_color = concat_points_with_color(
            self.must_monitor, self.get_color("must_monitor"), -EPSILON
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
        Visualize occupancy, camera pos, visible points and must_monitor.
        """

        XEPSILON = np.array([0.1, 0, 0])
        YEPSILON = np.array([0, 0.1, 0])
        occupancy_with_color = concat_points_with_color(
            self.occupancy, self.get_color("occupancy")
        )
        cameras_with_color = concat_points_with_color(
            self.cam_extrinsics, self.get_color("camera"), XEPSILON
        )
        visible_with_color = concat_points_with_color(
            [self.must_monitor[index] for index in self.visible_points],
            self.get_color("visible"),
            -XEPSILON,
        )
        must_monitor_with_color = concat_points_with_color(
            self.must_monitor, self.get_color("must_monitor"), YEPSILON
        )

        if visible_with_color is not None:
            points_with_color = np.row_stack(
                (
                    occupancy_with_color,
                    cameras_with_color,
                    visible_with_color,
                    must_monitor_with_color,
                ),
            )
        else:
            print(WARN("No visible point found."))
            points_with_color = np.row_stack(
                (
                    occupancy_with_color,
                    cameras_with_color,
                    must_monitor_with_color,
                ),
            )

        return points_with_color
