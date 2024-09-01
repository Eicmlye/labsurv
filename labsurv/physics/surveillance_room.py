import math
import os
import os.path as osp
import pickle
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from labsurv.utils.string import WARN
from labsurv.utils.surveillance import (
    COLOR_MAP,
    AdjustUninstalledCameraError,
    DeleteUninstalledCameraError,
    InstallAtExistingCameraError,
    build_block,
    compute_single_cam_visibility,
    concat_points_with_color,
    save_visualized_points,
    shift,
)
from mmcv import Config
from torch import Tensor


class SurveillanceRoom:
    INT = torch.int64
    FLOAT = torch.float16

    def __init__(
        self,
        device: Optional[torch.cuda.device],
        cfg_path: Optional[str] = None,
        shape: Optional[List[int]] = None,
        load_from: Optional[str] = None,
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
                "Either (`cfg_path`, `shape`) or `load_from` should be " "specified."
            )
        if cfg_path is not None and shape is not None and load_from is not None:
            print(
                WARN(
                    "Both (`cfg_path`, `shape`) and `load_from` are "
                    "specified. Only use `load_from`."
                )
            )

        self.device = device
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

            self.cfg_path: str = cfg_path
            self.shape: List[int] = shape
            self.occupancy: Tensor = torch.zeros(
                self.shape, dtype=self.INT, device=self.device
            )
            self.install_permitted: Tensor = torch.zeros(
                self.shape, dtype=self.INT, device=self.device
            )
            # need_monitor, h_res_req_min, h_res_req_max, v_res_req_min, v_res_req_max
            self.must_monitor: Tensor = torch.zeros(
                self.shape + [5], dtype=self.FLOAT, device=self.device
            )

            # is_installed, pan, tilt, cam_type
            self.cam_extrinsics: Tensor = torch.zeros(
                self.shape + [4], dtype=self.FLOAT, device=self.device
            )
            self.visible_points: Tensor = torch.zeros(
                self.shape, dtype=self.INT, device=self.device
            )
        else:
            if not osp.exists(load_from):
                raise ValueError(f"The path {load_from} does not exist.")
            if not load_from.endswith(".pkl"):
                raise ValueError("Only `pkl` files can be loaded.")

            with open(load_from, "rb") as fpkl:
                room_data = pickle.load(fpkl)
            self.cfg_path: str = room_data["cfg_path"]

            self.shape: List[int] = room_data["shape"]
            self.occupancy: Tensor = room_data["occupancy"]
            self.install_permitted: Tensor = room_data["install_permitted"]
            self.must_monitor: Tensor = room_data["must_monitor"]

            self.cam_extrinsics: Tensor = room_data["cam_extrinsics"]
            self.visible_points: Tensor = room_data["visible_points"]

        cfg = Config.fromfile(self.cfg_path)
        self._CAM_INTRINSICS: Dict = cfg.cam_intrinsics
        self._CAM_TYPES: List[str] = list(self._CAM_INTRINSICS.keys())
        self._POINT_CONFIGS: Dict[str, Dict[str, str | List[str]]] = cfg.point_configs
        self.voxel_length: float = cfg.voxel_length
        if not isinstance(self._CAM_INTRINSICS, dict):
            raise ValueError(
                "Loaded cam intrinsics should be a dict, "
                f"not {type(self._CAM_INTRINSICS)}."
            )

    def get_color(self, point_type: str) -> np.ndarray:
        """
        Get corresponding color RGB representation from a point type name string.
        """
        return COLOR_MAP[self._POINT_CONFIGS[point_type]["color"]]

    def get_extra_params_namelist(self, point_type: str) -> List[str]:
        return self._POINT_CONFIGS[point_type]["extra_params"]

    def _assert_dtype_device(
        self, tensor: Tensor, dtype, device: Optional[torch.cuda.device] = None
    ):
        assert dtype in [self.INT, self.FLOAT]
        if device is None:
            device = self.device

        assert (
            tensor.device == device
        ), f"Different devices found. Expected {device}, got {tensor.device}."
        assert tensor.dtype == dtype

    def _check_point_type(self, point_type: str) -> bool:
        """
        Check if the given point type is known.
        """
        return point_type in self._POINT_CONFIGS.keys()

    def _check_inside_room(self, points: Tensor) -> bool:
        """
        # Arguments:

            points (Tensor): input shape should be [3] or [n, 3].
        """
        dim = points.ndim
        assert dim == 1 or dim == 2, f"Cannot deal with {dim}-dimensional tensors."
        self._assert_dtype_device(points, self.INT)

        if dim == 1:
            points = [points]

        for point in points:
            assert point.numel() == 3, f"Invalid point coordinate {point}."

            try:
                self.occupancy[point[0], point[1], point[2]]
            except IndexError:
                return False

        return True

    def _check_permit_install(self, cam_pos: Tensor) -> bool:
        """
        # Arguments:

            cam_pos (Tensor): input shape should be [3] or [n, 3].
        """
        dim = cam_pos.ndim
        assert dim == 1 or dim == 2, f"Cannot deal with {dim}-dimensional tensors."
        self._assert_dtype_device(cam_pos, self.INT)

        if dim == 2:
            assert (
                cam_pos.shape[1] == 3
            ), f"Invalid point coordinate of length {cam_pos.shape[1]}."
            chosen_pos_permissions = self.install_permitted[
                cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2]
            ]
            return chosen_pos_permissions[chosen_pos_permissions == 0].numel() == 0

        # cam_pos.ndim == 1
        return self.install_permitted[cam_pos[0], cam_pos[1], cam_pos[2]]

    def _check_match_extra_params(
        self, point_type: str, **extra_params
    ) -> List[Set[str]] | Set[str]:
        """
        Check if the keys of the given param dict matches that of the required one.
        """
        provided = set(extra_params.keys())
        if "extra_params" not in self._POINT_CONFIGS[point_type].keys():
            if len(extra_params.keys()) == 0:
                return provided

            return [set(), provided]

        required = set(self._POINT_CONFIGS[point_type]["extra_params"])

        if required == provided:
            return provided

        return [required, provided]

    def add_block(
        self,
        shape: List[int],
        point_type: str = "occupancy",
        near_origin_vertex: Optional[Tensor] = None,
        **kwargs,
    ):
        # NOTE(eric): THIS METHOD SHOULD ONLY BE USED BEFORE ANY CAMERA OPERATION IS
        # MADE. THIS METHOD WILL NOT CHECK VISIBILITY CHANGES.
        if not self._check_point_type(point_type):
            raise ValueError(
                f"SurveillanceRoom does not support {point_type} type points."
            )

        if near_origin_vertex is None:
            near_origin_vertex = torch.tensor(
                [0, 0, 0], dtype=self.INT, device=self.device
            )
        else:
            self._assert_dtype_device(near_origin_vertex, self.INT)

        points = build_block(shape, self.device)
        if not self._check_inside_room(points):
            raise ValueError("Point outside of the room.")

        # consider `near_origin_vertex` as the displacement vector
        points = shift(points, near_origin_vertex, int_output=True)

        # NOTE(eric): `target_mask` is a pointer
        target_mask = None
        if point_type == "occupancy":
            target_mask = self.occupancy
        elif point_type == "install_permitted":
            target_mask = self.install_permitted
        elif point_type == "must_monitor":
            target_mask = self.must_monitor

        extra_params = self._check_match_extra_params(point_type, **kwargs)
        appendix: Optional[Tensor] = None
        if isinstance(extra_params, Set):
            if len(extra_params) > 0:
                try:
                    appendix = torch.tensor(
                        [
                            kwargs[key]
                            for key in self.get_extra_params_namelist(point_type)
                        ],
                        dtype=self.FLOAT,
                        device=self.device,
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

        if appendix is not None:
            full_appendix = torch.cat(
                (torch.tensor([1], dtype=self.FLOAT, device=self.device), appendix)
            ).repeat(points.shape[0], 1)

            target_mask[points[:, 0], points[:, 1], points[:, 2]] = full_appendix
        else:
            target_mask[points[:, 0], points[:, 1], points[:, 2]] = 1

    def add_cam(self, pos: Tensor, direction: Tensor, cam_type: str | int):
        """
        # Arguments:

            pos (Tensor): the position of the camera. Must be on the room points.

            direction (Tensor): the pan and tilt angle of the camera. pan in
            [-pi, pi), tilt in [-pi/2, pi/2].

            cam_type (str | int): the type (index) of camera.
        """
        assert pos.ndim == 1 and pos.numel() == 3
        self._assert_dtype_device(pos, self.INT)
        if not self._check_inside_room(pos):
            raise ValueError("Point outside of the room.")
        if not self._check_permit_install(pos):
            raise ValueError(f"{pos} is not permitted to install.")
        if self.cam_extrinsics[pos[0], pos[1], pos[2]][0] == 1:
            raise InstallAtExistingCameraError(f"Existed camera found at {pos}.")

        self._assert_dtype_device(direction, self.FLOAT)
        assert direction.ndim == 1 and direction.numel() == 2
        assert direction[0] >= -math.pi and direction[0] < math.pi
        assert direction[1] >= -math.pi / 2 and direction[1] <= math.pi / 2

        if isinstance(cam_type, int):
            if cam_type >= len(self._CAM_TYPES):
                raise ValueError(
                    f"SurveillanceRoom supports {len(self._CAM_TYPES)} "
                    f"types of cameras. Cam Index {cam_type} is invalid."
                )
        elif cam_type not in self._CAM_TYPES:
            raise ValueError(
                f"SurveillanceRoom does not support \"{cam_type}\" cameras."
            )
        else:
            cam_type = self._CAM_TYPES.index(cam_type)

        extrinsics = torch.cat(
            (
                torch.tensor([1], dtype=self.FLOAT, device=self.device),
                direction,
                torch.tensor([cam_type], dtype=self.FLOAT, device=self.device),
            )
        )
        self.cam_extrinsics[pos[0], pos[1], pos[2]] = extrinsics

        vis_mask = compute_single_cam_visibility(
            pos,
            direction,
            self._CAM_INTRINSICS[self._CAM_TYPES[cam_type]],
            self.occupancy,
            self.must_monitor,
            self.voxel_length,
        )

        self.visible_points += vis_mask

    def del_cam(self, pos: Tensor):
        """
        # Arguments:

            pos (Tensor): the position of the camera. Must be on the room points.
        """
        assert pos.ndim == 1 and pos.numel() == 3
        self._assert_dtype_device(pos, self.INT)
        if not self._check_inside_room(pos):
            raise ValueError("Point outside of the room.")
        if self.cam_extrinsics[pos[0], pos[1], pos[2]][0] == 0:
            raise DeleteUninstalledCameraError("Cannot delete uninstalled camera.")

        extrinsics = torch.cat(
            (
                torch.tensor([0], dtype=self.FLOAT, device=self.device),
                self.cam_extrinsics[pos[0], pos[1], pos[2]][1:],
            )
        )
        self.cam_extrinsics[pos[0], pos[1], pos[2]] = extrinsics

        vis_mask = compute_single_cam_visibility(
            pos,
            extrinsics[1:3],
            self._CAM_INTRINSICS[self._CAM_TYPES[extrinsics[-1]]],
            self.occupancy,
            self.must_monitor,
            self.voxel_length,
        )

        self.visible_points -= vis_mask

    def adjust_cam(
        self, pos: Tensor, direction: Tensor, cam_type: str | int | None = None
    ):
        """
        # Arguments:

            pos (Tensor): the position of the camera. Must be on the room points.

            direction (Tensor): the pan and tilt angle of the camera. pan in
            [-pi, pi), tilt in [-pi/2, pi/2].

            cam_type (str | int | None): the type (index) of camera. If is None,
            cam_type will not change.
        """
        assert pos.ndim == 1 and pos.numel() == 3
        self._assert_dtype_device(pos, self.INT)
        if not self._check_inside_room(pos):
            raise ValueError("Point outside of the room.")
        if not self._check_permit_install(pos):
            raise ValueError(f"{pos} is not permitted to install.")
        if self.cam_extrinsics[pos[0], pos[1], pos[2]][0] == 0:
            raise AdjustUninstalledCameraError("Cannot adjust uninstalled camera.")

        self._assert_dtype_device(direction, self.FLOAT)
        assert direction.ndim == 1 and direction.numel() == 2
        assert direction[0] >= -math.pi and direction[0] < math.pi
        assert direction[1] >= -math.pi / 2 and direction[1] <= math.pi / 2
        direction = direction.to(self.device)

        if isinstance(cam_type, int):
            if cam_type >= len(self._CAM_TYPES):
                raise ValueError(
                    f"SurveillanceRoom supports {len(self._CAM_TYPES)} "
                    f"types of cameras. Cam Index {cam_type} is invalid."
                )
        elif cam_type not in self._CAM_TYPES:
            raise ValueError(
                f"SurveillanceRoom does not support \"{cam_type}\" cameras."
            )
        else:
            cam_type = self._CAM_TYPES.index(cam_type)

        cam_type = torch.tensor([cam_type], dtype=self.FLOAT, device=self.device)

        pred_extrinsics = self.cam_extrinsics[pos[0], pos[1], pos[2]]
        pred_vis_mask = compute_single_cam_visibility(
            pos,
            pred_extrinsics[1:3],
            self._CAM_INTRINSICS[self._CAM_TYPES[pred_extrinsics[-1]]],
            self.occupancy,
            self.must_monitor,
            self.voxel_length,
        )
        self.visible_points -= pred_vis_mask

        extrinsics = torch.cat(
            (
                torch.tensor([1], dtype=self.FLOAT, device=self.device),
                direction,
                cam_type,
            )
        )
        self.cam_extrinsics[pos[0], pos[1], pos[2]] = extrinsics

        pred_vis_mask = compute_single_cam_visibility(
            pos,
            direction,
            self._CAM_INTRINSICS[self._CAM_TYPES[cam_type]],
            self.occupancy,
            self.must_monitor,
            self.voxel_length,
        )
        self.visible_points += pred_vis_mask

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

    def visualize_occupancy(self) -> Tensor:
        """
        Visualize occupancy, install_permitted and must_monitor.
        """

        EPSILON = torch.tensor([0.1, 0, 0], device=self.device)
        occupancy_with_color = concat_points_with_color(
            self.occupancy, self.get_color("occupancy")
        )
        install_permitted_with_color = concat_points_with_color(
            self.install_permitted, self.get_color("install_permitted")  # , EPSILON
        )
        must_monitor_with_color = concat_points_with_color(
            self.must_monitor, self.get_color("must_monitor")  # , -EPSILON
        )

        points_with_color = torch.cat(
            (  # points rendered later will cover the earlier ones.
                must_monitor_with_color,
                occupancy_with_color,
                install_permitted_with_color,
            ),
            0,
        )

        return points_with_color

    def visualize_camera(self) -> Tensor:
        """
        Visualize occupancy, camera pos, visible points and must_monitor.
        """

        XEPSILON = torch.tensor([0.1, 0, 0], device=self.device)
        YEPSILON = torch.tensor([0, 0.1, 0], device=self.device)
        occupancy_with_color = concat_points_with_color(
            self.occupancy, self.get_color("occupancy")
        )
        cameras_with_color = concat_points_with_color(
            self.cam_extrinsics, self.get_color("camera")  # , XEPSILON
        )
        visible_with_color = concat_points_with_color(
            self.visible_points, self.get_color("visible")  # , -XEPSILON
        )
        must_monitor_with_color = concat_points_with_color(
            self.must_monitor, self.get_color("must_monitor")  # , YEPSILON
        )

        if visible_with_color is not None:
            points_with_color = torch.cat(
                (  # points rendered later will cover the earlier ones.
                    # must_monitor_with_color,
                    occupancy_with_color,
                    cameras_with_color,
                    visible_with_color,
                ),
                0,
            )
        else:
            print(WARN("No visible point found."))
            points_with_color = torch.cat(
                (  # points rendered later will cover the earlier ones.
                    must_monitor_with_color,
                    occupancy_with_color,
                    cameras_with_color,
                ),
                0,
            )

        return points_with_color
