import os.path as osp
import pickle
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from labsurv.utils.string import WARN, to_filename
from labsurv.utils.surveillance import (
    COLOR_MAP,
    AdjustUninstalledCameraError,
    DeleteUninstalledCameraError,
    InstallAtExistingCameraError,
    apply_colormap_to_list,
    build_block,
    compute_single_cam_visibility,
    concat_points_with_color,
    save_visualized_points,
    shift,
)
from matplotlib.colors import LinearSegmentedColormap
from mmcv import Config, ProgressBar
from numpy import ndarray as array
from torch import Tensor


class SurveillanceRoom:
    """
    ## Description:

        The `SurveillanceRoom` class maintains geometric data and camera settings of
        the room interior. It is not necessary to be a SINGLE room. One may make it a
        large region of interest and add blocks to separate different rooms.

    ## Attributes:

        INT: synonyms for torch.int64.

        FLOAT: synonyms for torch.float.

        AINT: synonyms for np.int64.

        AFLOAT: synonyms for np.float32.

        cam_modify_num (int): number of camera operations have been made.

        device (torch.cuda_device): the device all the tensors are stored.

        cfg_path (Optional[str]): the configuration file path of the clip size,
        focal length and other parameters of the cameras and the room.

        shape (List[int]): [3], the shape of the room.

        occupancy (Tensor): [W, D, H], torch.int64, the boolean mask of the occupancy.

        install_permitted (Tensor): [W, D, H], torch.int64, the boolean mask of the
        installation permission.

        must_monitor (Tensor): [W, D, H, 5], torch.float,
        [need_monitor, h_res_req_min, h_res_req_max, v_res_req_min, v_res_req_max].
        [:,:,:, 0] is a mask indicating if the position should be monitored,
        [:,:,:, 1:] represent the horizontal/vertical resolution requirements for
        current position.

        cam_extrinsics (Tensor): [W, D, H, 4], torch.float,
        [is_installed, pan, tilt, cam_type].
        [:,:,:, 0] is a mask indicating if any camera is installed at this position,
        [:,:,:, 1:] are the orientation and cam type of the cameras installed.

        visible_points (Tensor): [W, D, H], torch.int64, visibility mask. The value
        represents the number of cameras that watching this voxel.
    """

    INT = torch.int64
    FLOAT = torch.float
    AINT = np.int64
    AFLOAT = np.float32

    def __init__(
        self,
        device: Optional[torch.cuda.device],
        cfg_path: Optional[str] = None,
        shape: Optional[List[int]] = None,
        load_from: Optional[str] = None,
    ):
        """
        ## Argument:

            device (Optional[torch.cuda.device]): the device where all the tensors are
            placed.

            cfg_path (Optional[str]): the configuration file path of the clip size,
            focal length and other parameters of the cameras and the room.

            shape (Optional[List[int]]): if not `None`, a room of `shape` size will
            be generated. Physically, the room is of shape `voxel_length * shape`,
            where the voxel centers are treated as points in the room.

            load_from (Optional[str]): only valid if `cfg_path` and `shape` are `None`,
            and the room will be loaded from `load_from`.
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

        self.device = device
        self.cam_modify_num: int = 0
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
        self._CAM_INTRINSICS: Dict[str, Dict[str, Dict[str, float | List[float]]]] = (
            cfg.cam_intrinsics
        )
        self._CAM_TYPES: List[str] = list(self._CAM_INTRINSICS.keys())
        self._POINT_CONFIGS: Dict[str, Dict[str, str | List[str]]] = cfg.point_configs
        self.voxel_length: float = cfg.voxel_length

    def get_color(self, point_type: str) -> array:
        """
        ## Description:

            Get corresponding color RGB representation from a point type name string.
        """
        return COLOR_MAP[self._POINT_CONFIGS[point_type]["color"]]

    def get_extra_params_namelist(self, point_type: str) -> List[str]:
        return self._POINT_CONFIGS[point_type]["extra_params"]

    def get_cam_type_index(self, cam_type: str | int) -> int:
        if isinstance(cam_type, (int, self.AINT)) or (
            isinstance(cam_type, torch.Tensor)
            and cam_type.dtype == self.INT
            and cam_type.ndim == 1
            and cam_type.shape[0] == 1
        ):
            if cam_type >= len(self._CAM_TYPES):
                raise ValueError(
                    f"SurveillanceRoom supports {len(self._CAM_TYPES)} "
                    f"types of cameras. Cam Index {cam_type} is invalid."
                )
        elif isinstance(cam_type, str):
            if cam_type not in self._CAM_TYPES:
                raise ValueError(
                    f"SurveillanceRoom does not support \"{cam_type}\" cameras."
                )
            else:
                cam_type = self._CAM_TYPES.index(cam_type)
        else:
            raise ValueError(
                f"Unsupported cam_type {cam_type} in type {type(cam_type)}"
            )

        return cam_type

    def get_cam_intrinsics(self, cam_type: str | int):
        cam_type_index = self.get_cam_type_index(cam_type)

        return self._CAM_INTRINSICS[self._CAM_TYPES[cam_type_index]]

    def _assert_dtype_device(
        self, tensor: Tensor, dtype, device: Optional[torch.cuda.device] = None
    ):
        """
        ## Description:

            Go through assertions of dtype and device for a tensor.
        """
        assert dtype in [self.INT, self.FLOAT]
        if device is None:
            device = self.device

        assert (
            str(tensor.device).startswith("cuda") and str(device).startswith("cuda")
        ) or (
            str(tensor.device).startswith("cpu") and str(device).startswith("cpu")
        ), f"Different devices found. Expected {device}, got {tensor.device}."
        assert tensor.dtype == dtype

    def _check_point_type(self, point_type: str) -> bool:
        """
        ## Description:

            Check if the given point type is known.
        """
        return point_type in self._POINT_CONFIGS.keys()

    def _check_inside_room(self, points: Tensor) -> bool:
        """
        ## Arguments:

            points (Tensor): [3] or [N, 3], torch.int64.
        """
        dim = points.ndim
        assert dim == 1 or dim == 2, f"Cannot deal with {dim}-dimensional tensors."
        self._assert_dtype_device(points, self.INT)

        if dim == 1:
            points = torch.unsqueeze(points, 0)
        assert points.shape[1] == 3

        try:
            self.occupancy[points[:, 0], points[:, 1], points[:, 2]]
        except IndexError:
            return False

        return True

    def _check_permit_install(self, cam_pos: Tensor) -> bool:
        """
        ## Arguments:

            cam_pos (Tensor): [3] or [N, 3], torch.int64.
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
        ## Description:

            Check if the keys of the given param dict matches that of the required one.

        ## Returns:

            a set if matches, a list otherwise.
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
        displacement: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        ## Arguments:

            shape (List[int]): [3], the shape of the block.

            point_type (str): the type of the point, chosen from cfg file.

            displacement (Optional[List[int]]): [3], the displacement of the
            block, consider the block is geenrated from the origin. If None, set
            default to [0, 0, 0].

            kwargs: extra_param dict, must match `point_type`.
        """
        # NOTE(eric): THIS METHOD SHOULD ONLY BE USED BEFORE ANY CAMERA OPERATION IS
        # MADE. THIS METHOD WILL NOT CHECK VISIBILITY CHANGES.
        assert (
            self.cam_modify_num == 0
        ), "Adding blocks after modifying cameras could result in visibility error."

        if not self._check_point_type(point_type):
            raise ValueError(
                f"`SurveillanceRoom` does not support {point_type} type points."
            )

        if displacement is None:
            displacement = [0, 0, 0]
        displacement = torch.tensor(displacement, dtype=self.INT, device=self.device)

        points = build_block(shape, self.device)
        if not self._check_inside_room(points):
            raise ValueError("Point outside of the room.")

        points = shift(points, displacement, int_output=True)

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

    def add_cam(
        self,
        pos: array,
        direction: array,
        cam_type: str | int,
        provided_vismask: Optional[Tensor] = None,
    ) -> array:
        """
        ## Arguments:

            pos (np.ndarray): [3], np.int64, the position of the camera.

            direction (np.ndarray): [2], np.float32, the `pan` and `tilt` angle of the
            camera. `pan` in [-pi, pi), `tilt` in [-pi/2, pi/2].

            cam_type (str | int): the type (index) of camera.

            provided_vismask (Optional[Tensor]): if provided, the vismask will be used
            directly to speedup computation, lov arguments will be ignored and the
            correctness of this vismask should be promised.

        ## Returns:

            vis_mask (np.ndarray): the visibility mask of the added camera.
        """
        self.cam_modify_num += 1

        pos = torch.tensor(pos, dtype=self.INT, device=self.device)
        assert pos.ndim == 1 and pos.numel() == 3
        if not self._check_inside_room(pos):
            raise ValueError("Point outside of the room.")
        if not self._check_permit_install(pos):
            raise ValueError(f"{pos} is not permitted to install.")
        if self.cam_extrinsics[pos[0], pos[1], pos[2]][0] == 1:
            raise InstallAtExistingCameraError(f"Existed camera found at {pos}.")

        direction = torch.tensor(direction, dtype=self.FLOAT, device=self.device)
        assert direction.ndim == 1 and direction.numel() == 2
        assert direction[0] >= -torch.pi and direction[0] < torch.pi
        assert direction[1] >= -torch.pi / 2 and direction[1] <= torch.pi / 2

        cam_type_index = self.get_cam_type_index(cam_type)

        extrinsics = torch.cat(
            (
                torch.tensor([1], dtype=self.FLOAT, device=self.device),
                direction,
                torch.tensor([cam_type_index], dtype=self.FLOAT, device=self.device),
            )
        )
        self.cam_extrinsics[pos[0], pos[1], pos[2]] = extrinsics

        if provided_vismask is None:
            vis_mask = compute_single_cam_visibility(
                pos,
                direction,
                self.get_cam_intrinsics(cam_type_index),
                self.occupancy,
                self.must_monitor,
                self.voxel_length,
            )
        else:
            vis_mask = provided_vismask

        self.visible_points += vis_mask

        return vis_mask.cpu().numpy().copy()

    def del_cam(
        self,
        pos: array,
        provided_vismask: Optional[Tensor] = None,
    ) -> array:
        """
        ## Arguments:

            pos (np.ndarray): [3], np.int64, the position of the camera.

            provided_vismask (Optional[Tensor]): if provided, the vismask will be used
            directly to speedup computation, lov arguments will be ignored and the
            correctness of this vismask should be promised.

        ## Returns:

            vis_mask (np.ndarray): the visibility mask of the deleted camera.
        """
        self.cam_modify_num += 1

        pos = torch.tensor(pos, dtype=self.INT, device=self.device)
        assert pos.ndim == 1 and pos.numel() == 3
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

        if provided_vismask is None:
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
                self._CAM_INTRINSICS[self._CAM_TYPES[extrinsics[-1].type(self.INT)]],
                self.occupancy,
                self.must_monitor,
                self.voxel_length,
            )
        else:
            vis_mask = provided_vismask

        self.visible_points -= vis_mask

        return vis_mask.cpu().numpy().copy()

    def adjust_cam(
        self,
        pos: array,
        direction: array,
        cam_type: Optional[str | int] = None,
        provided_pred_vismask: Optional[Tensor] = None,
        provided_vismask: Optional[Tensor] = None,
        delta_vismask_out: bool = False,
    ) -> array | Tuple[array, array]:
        """
        ## Arguments:

            pos (np.ndarray): [3], np.int64, the position of the camera.

            direction (np.ndarray): [2], np.float32, the `pan` and `tilt` angle of the
            camera. `pan` in [-pi, pi), `tilt` in [-pi/2, pi/2].

            cam_type (Optional[str | int]): the type (index) of camera. If is None,
            `cam_type` will not change.

            provided_pred_vismask (Optional[Tensor]): this vismask is used for del. If
            provided, the vismask will be used directly to speedup computation, lov
            arguments will be ignored and the correctness of this vismask should be
            promised.

            provided_vismask (Optional[Tensor]): this vismask is used for add. If
            provided, the vismask will be used directly to speedup computation, lov
            arguments will be ignored and the correctness of this vismask should be
            promised.

            delta_vismask_out (bool): whether return two vismasks separately (False) or
            just return the delta vismask (True).

        ## Returns:

            vis_mask (np.ndarray | Tuple[np.ndarray, np.ndarray]): the visibility mask
            of the modified camera.
        """
        self.cam_modify_num += 1

        pos = torch.tensor(pos, dtype=self.INT, device=self.device)
        assert pos.ndim == 1 and pos.numel() == 3
        if not self._check_inside_room(pos):
            raise ValueError("Point outside of the room.")
        if not self._check_permit_install(pos):
            raise ValueError(f"{pos} is not permitted to install.")
        if self.cam_extrinsics[pos[0], pos[1], pos[2]][0] == 0:
            raise AdjustUninstalledCameraError("Cannot adjust uninstalled camera.")

        direction = torch.tensor(direction, dtype=self.FLOAT, device=self.device)
        assert direction.ndim == 1 and direction.numel() == 2
        assert direction[0] >= -torch.pi and direction[0] < torch.pi
        assert direction[1] >= -torch.pi / 2 and direction[1] <= torch.pi / 2
        direction = direction.to(self.device)

        pred_extrinsics = self.cam_extrinsics[pos[0], pos[1], pos[2]]
        pred_cam_type_index = pred_extrinsics[-1].type(self.INT).item()

        cam_type_index = (
            pred_cam_type_index
            if cam_type is None
            else self.get_cam_type_index(cam_type)
        )

        if provided_pred_vismask is None:
            pred_vis_mask = compute_single_cam_visibility(
                pos,
                pred_extrinsics[1:3],
                self._CAM_INTRINSICS[self._CAM_TYPES[pred_cam_type_index]],
                self.occupancy,
                self.must_monitor,
                self.voxel_length,
            )
        else:
            pred_vis_mask = provided_pred_vismask
        self.visible_points -= pred_vis_mask

        extrinsics = torch.cat(
            (
                torch.tensor([1], dtype=self.FLOAT, device=self.device),
                direction,
                torch.tensor([cam_type_index], dtype=self.FLOAT, device=self.device),
            )
        )
        self.cam_extrinsics[pos[0], pos[1], pos[2]] = extrinsics

        if provided_vismask is None:
            vis_mask = compute_single_cam_visibility(
                pos,
                direction,
                self.get_cam_intrinsics(cam_type_index),
                self.occupancy,
                self.must_monitor,
                self.voxel_length,
            )
        else:
            vis_mask = provided_vismask
        self.visible_points += vis_mask

        if delta_vismask_out:
            return (vis_mask - pred_vis_mask).cpu().numpy().copy()
        else:
            return pred_vis_mask.cpu().numpy().copy(), vis_mask.cpu().numpy().copy()

    def save(self, save_path: str):
        """
        ## Description:

            Saves the `SurveillanceRoom` class data in a `pkl` file.
        """
        save_pkl_path = to_filename(save_path, ".pkl", "SurveillanceRoom")

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

    def visualize(self, save_path: str, mode: str = "occupancy", heatmap: bool = False):
        """
        ## Description:

            Saves the pointcloud in a `ply` file.

        ## Arguments:

            save_path (str): if ends with `.ply`, the exact file name will be used,
            otherwise the path is treated as a directory and `SurveillanceRoom.ply` is
            created inside this directory.

            mode (str): specifying what to visualize. Supported mode: `occupancy`,
            `camera`.
        """

        if mode == "occupancy":
            points_with_color = self._visualize_occupancy()
        elif mode == "camera":
            points_with_color = self._visualize_camera(heatmap)
        else:
            raise ValueError(f"Unsupported mode {mode}.")

        if points_with_color is not None:
            save_visualized_points(
                points_with_color, save_path, "SurveillanceRoom_" + mode[:3]
            )

    def _visualize_occupancy(self) -> Tensor:
        """
        ## Desctiption:

            Visualize occupancy, install_permitted and must_monitor.

        ## Returns:

            points_with_color (Tensor): [N, 6], [x, y, z, R, G, B], torch.float16.
        """

        # EPSILON = torch.tensor([0.1, 0, 0], device=self.device)
        occupancy_with_color = concat_points_with_color(
            self.occupancy, self.get_color("occupancy")
        )
        install_permitted_with_color = concat_points_with_color(
            self.install_permitted, self.get_color("install_permitted")  # , EPSILON
        )
        must_monitor_with_color = concat_points_with_color(
            self.must_monitor, self.get_color("must_monitor")  # , -EPSILON
        )

        to_be_visualized = (
            (  # points rendered later will cover the earlier ones.
                must_monitor_with_color,
                occupancy_with_color,
                install_permitted_with_color,
            )
            if occupancy_with_color is not None
            else (
                must_monitor_with_color,
                install_permitted_with_color,
            )
        )
        points_with_color = torch.cat(to_be_visualized, 0)

        return points_with_color

    def _visualize_camera(self, heatmap: bool = False) -> Tensor:
        """
        ## Description:

            Visualize occupancy, camera pos, visible points and must_monitor.

        ## Returns:

            points_with_color (Tensor): [N, 6], [x, y, z, R, G, B], torch.float16.
        """

        # XEPSILON = torch.tensor([0.1, 0, 0], device=self.device)
        # YEPSILON = torch.tensor([0, 0.1, 0], device=self.device)
        occupancy_with_color = concat_points_with_color(
            self.occupancy, self.get_color("occupancy")
        )
        cameras_with_color = concat_points_with_color(
            self.cam_extrinsics, self.get_color("camera")  # , XEPSILON
        )

        if cameras_with_color is None:
            print(WARN("No cameras have been placed. Visualization aborted."))
            return None

        if heatmap and self.visible_points.sum() > 0:
            CENTER_SHIFT = torch.tensor(
                [0.5, 0.5, 0.5], dtype=torch.float16, device=self.device
            )
            colors = [
                "#000000",  # black, min
                "#00ff00",  # green, max
            ]
            colormap = LinearSegmentedColormap.from_list("custom", colors, N=256)
            colored_dist: Tensor = apply_colormap_to_list(  # [N, 3]
                self.visible_points[self.visible_points > 0]
                .view(-1)
                .cpu()
                .numpy()
                .copy(),
                colormap=colormap,
                device=torch.device("cuda"),
                divide_by_max=True,
            )

            visible_with_color = torch.cat(
                (shift(self.visible_points.nonzero(), CENTER_SHIFT), colored_dist),
                dim=1,
            )  # [N, 6]
        else:
            visible_with_color = concat_points_with_color(
                self.visible_points, self.get_color("visible")  # , -XEPSILON
            )
        must_monitor_with_color = concat_points_with_color(
            self.must_monitor, self.get_color("must_monitor")  # , YEPSILON
        )

        if visible_with_color is not None:
            to_be_visualized = (
                (  # points rendered later will cover the earlier ones.
                    occupancy_with_color,
                    cameras_with_color,
                    visible_with_color,
                )
                if occupancy_with_color is not None
                else (
                    cameras_with_color,
                    visible_with_color,
                )
            )
            points_with_color = torch.cat(to_be_visualized, 0)
        else:
            print(WARN("No visible point found."))
            to_be_visualized = (
                (  # points rendered later will cover the earlier ones.
                    must_monitor_with_color,
                    occupancy_with_color,
                    cameras_with_color,
                )
                if occupancy_with_color is not None
                else (
                    must_monitor_with_color,
                    cameras_with_color,
                )
            )
            points_with_color = torch.cat(to_be_visualized, 0)

        return points_with_color

    def get_info(self) -> array:
        """
        ## Description:

            Merge all the tensor attributes to a single array. Usually used to avoid
            gpu memory leaks.

        ## Returns:

            result (np.ndarray): [12, W, D, H], np.float32.
        """

        # .type method always return a new tensor
        result = (
            torch.cat(
                (
                    self.occupancy.type(self.FLOAT).unsqueeze(0),  # [1, W, D, H]
                    self.install_permitted.type(self.FLOAT).unsqueeze(
                        0
                    ),  # [1, W, D, H]
                    self.must_monitor.type(self.FLOAT).permute(
                        3, 0, 1, 2
                    ),  # [5, W, D, H]
                    self.cam_extrinsics.type(self.FLOAT).permute(
                        3, 0, 1, 2
                    ),  # [4, W, D, H]
                    self.visible_points.type(self.FLOAT).unsqueeze(0),  # [1, W, D, H]
                )
            )
            .cpu()
            .numpy()
            .copy()
        )

        return result

    def best_installation(
        self,
        pos: array,
        direction: List[float] | array,
        section_nums: List[int],
        cam_type: int,
    ) -> Tuple[int, float, List[List[float]], List[array], List]:
        """
        ## Description:

            Find the installation directions that increases the most coverage at `pos`
            with `cam_type`.

        ## Arguments:

            pos (np.ndarray): [3], np.int64, the pos coord.

            direction (List[float] | array): [2], float | np.float32, the pan and tilt
            angle values. This entry is only used to locate `input_direction_index`.

            section_nums (List[int]): [2], pan and tilt section nums.

            cam_type (int): camera type index.

        ## Returns:

            input_direction_index (int): the index of the transition correspond to the
            input direction.

            best_coverage_increment (float): The coverage increment when best installed.

            pan_tilt_list (List[List[float]]): [N, 2], list of all pan-tilt combinations.

            room_info_list (List[np.ndarray]): [N], list of room info array to all
            pan-tilt combinations.

            similarity_list (List[float]): [N], list of all similarity value to all
            pan-tilt combinations.
        """
        pan_list = [
            -np.pi + 2 * np.pi / section_nums[0] * k for k in range(section_nums[0])
        ]
        tilt_list = [
            -np.pi / 2 + np.pi / section_nums[1] * k for k in range(section_nums[1])
        ]

        total_target_point_num: float = self.must_monitor[:, :, :, 0].sum().item()
        pred_coverage: float = (
            self.visible_points > 0
        ).sum().item() / total_target_point_num

        best_coverage_increment: float = 0
        best_parameters = []

        print("\nComputing best installation params...")
        prog_bar = ProgressBar(len(pan_list) * (len(tilt_list) - 1) + 1)

        pan_tilt_list: List[List[float]] = []
        room_info_list: List[array] = []
        direction_coord_list: List[array] = []
        input_direction_index = -1
        index_counter = -1

        for tilt in tilt_list:
            for pan in pan_list:
                index_counter += 1
                if (
                    np.abs(pan - direction[0]) < 1e-6
                    and np.abs(tilt - direction[1]) < 1e-6
                ):
                    input_direction_index = index_counter

                cache_room: SurveillanceRoom = deepcopy(self)
                cache_room.add_cam(pos, np.array([pan, tilt]), cam_type)
                cur_coverage: float = (
                    cache_room.visible_points > 0
                ).sum().item() / total_target_point_num

                cov_incre = cur_coverage - pred_coverage

                if cov_incre > best_coverage_increment:
                    best_coverage_increment = cov_incre
                    best_parameters = [[pan, tilt]]
                elif cov_incre == best_coverage_increment:
                    best_parameters.append([pan, tilt])

                pan_tilt_list.append([pan, tilt])
                room_info_list.append(cache_room.get_info())
                direction_coord_list.append(
                    _pan_tilt_2_coord([pan, tilt], array_out=True)
                )

                prog_bar.update()
                if np.abs(tilt - (-np.pi / 2)) < 1e-6:
                    # deal with polar point
                    break
        print("\r\033[K\033[1A\033[K\033[1A\033[K\033[1A")

        best_vecs_list = []
        for param in best_parameters:
            best_vecs_list.append(_pan_tilt_2_coord(param))
        best_vecs: array = np.array(best_vecs_list)

        avg_vec = np.mean(best_vecs, axis=0)
        normalized_avg_vec = avg_vec / np.linalg.norm(avg_vec)

        similarity_list: List[float] = []
        for coord in direction_coord_list:
            similarity_list.append(np.sum(coord * normalized_avg_vec))

        return (
            input_direction_index,
            best_coverage_increment,
            pan_tilt_list,
            room_info_list,
            similarity_list,
        )


def _pan_tilt_2_coord(
    direction: List[float] | array, array_out: bool = False
) -> array | List[float]:
    """
    ## Description:

        Transform a [pan, tilt] direction to the corresponding directional vector
        [x, y, z].

    ## Arguments:

        direction (List[float] | np.ndarray): [2], pan and tilt angle in radian.

        array_out (bool): if transform the output to an array.

    ## Returns:

        coord (np.ndarray | List[float]): [3], the direction vector in xyz-coord.
    """
    assert len(direction) == 2

    pan, tilt = direction

    vec = [
        np.cos(tilt) * np.cos(pan),
        np.cos(tilt) * np.sin(pan),
        np.sin(tilt),
    ]
    return np.array(vec) if array_out else vec
