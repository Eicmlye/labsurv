import math
from typing import List

import numpy as np
from scipy.optimize import linprog


def compute_single_cam_visibility(
    extrinsic: np.ndarray,
    intrinsic: dict,
    occupancy: List[np.ndarray],
    targets: List[np.ndarray],
    voxel_length: float,
):
    """
    ## Arguments:

        extrinsic (np.ndarray): array([x, y, z, pan, tilt]), the position and
        orientation of the camera.

        intrinsic (dict): {clip_shape=[],focal_length=,resolution=[]}, the intrinsic of
        camera.

        occupancy (List[np.ndarray]): [array([x, y, z]), ...], the coordinates of the
        points occupied by objects.

        targets (List[np.ndarray]):
        [array([x, y, z, h_res_req_min/max, v_res_req_min/max]), ...], the
        coordinates of the target points along with the horizontal and vertical pixel
        resolution requirements at this position.

        voxel_length (float): the length of sides of pixels in meters.

    ## Returns:

        vis_dict (dict): {target_coord: np.ndarray=is_visible: bool}, if the target
        is visible to the camera.
    """

    cam_pos = extrinsic[:3]
    pan, tilt = extrinsic[3], extrinsic[4]
    clip_shape = intrinsic["clip_shape"]  # horizontal, vertical
    f = intrinsic["focal_length"]
    resolution = intrinsic["resolution"]  # horizontal, vertical

    rot_mat = np.array(
        [
            [math.cos(pan), 0, -math.sin(pan)],
            [0, 1, 0],
            [math.sin(pan), 0, math.cos(pan)],
        ]
    ) @ np.array(
        [
            [math.cos(tilt), 0, -math.sin(tilt)],
            [0, 1, 0],
            [math.sin(tilt), 0, math.cos(tilt)],
        ]
    )

    # angle of view
    h_aov_half = math.atan(clip_shape[0] / (2 * f))
    v_aov_half = math.atan(clip_shape[1] / (2 * f))

    vis_dict = dict()

    for index, target in enumerate(targets):
        target_pos = target[:3]
        h_res_req_min = target[3]
        h_res_req_max = target[4]
        v_res_req_min = target[5]
        v_res_req_max = target[6]

        # shape of farther view plane
        h_far = resolution[0] / h_res_req_min
        v_far = resolution[1] / v_res_req_min

        # shape of nearer view plane
        h_near = resolution[0] / h_res_req_max
        v_near = resolution[1] / v_res_req_max

        # depth of view
        d_far = min(f * h_far / clip_shape[0], f * v_far / clip_shape[1])
        d_near = min(f * h_near / clip_shape[0], f * v_near / clip_shape[1])

        # light_of_view
        lov = target_pos - cam_pos
        lov_cam_coord = rot_mat.T @ lov  # transform lov to cam coord

        vis_dict[index] = (
            check_inside_aov(lov_cam_coord, h_aov_half, v_aov_half)
            and check_inside_dof(lov_cam_coord, d_near, d_far, voxel_length)
            and check_obstacle(cam_pos, lov, occupancy)
        )

    return vis_dict


def check_inside_aov(lov_cam_coord: np.ndarray, h_aov_half: float, v_aov_half: float):
    """
    ## Description:

        Check if the target point is in the aov of the camera.

    ## Arguments:

        lov (np.ndarray): the vector from camera pos to target pos.

        h_aov_half (float): half horizontal aov of the camera.

        v_aov_half (float): half verticle aov of the camera.
    """
    assert h_aov_half < math.pi / 2
    assert v_aov_half <= math.pi / 2
    if np.array_equal(lov_cam_coord, np.array([0, 0, 0])):
        return False

    lov_xy_length = np.linalg.norm(lov_cam_coord[0:1])

    lov_h_angle = math.acos(lov_cam_coord[0] / lov_xy_length)
    if h_aov_half < lov_h_angle:
        return False

    lov_v_angle = (
        math.atan(lov_cam_coord[2] / lov_xy_length)
        if lov_xy_length != 0
        else math.pi / 2
    )
    return v_aov_half >= lov_v_angle


def check_inside_dof(
    lov_cam_coord: np.ndarray, d_near: float, d_far: float, voxel_length: float
):
    """
    ## Description:

        Check if the target point is in the dof of the camera.

    ## Arguments:

        lov (np.ndarray): the vector from camera pos to target pos.

        d_near (float): nearer dof of the camera in meters.

        d_far (float): farther dof of the camera in meters.

        voxel_length (float): the length of sides of pixels in meters.
    """
    d_lov = lov_cam_coord[0] * voxel_length

    return d_lov >= d_near and d_lov <= d_far


def check_obstacle(cam_pos: np.ndarray, lov: np.ndarray, occupancy: List[np.ndarray]):
    """
    ## Description:

        Check if the light of view is block by any obstacles.

        This is an linear programming problem for all the voxels. Let the camera pos be
        `c` and light of view be `lov`, the problem is to find whether a `lambda`
        satisfying the following inequations exists:

        `c + (1 - lambda) * lov in voxel`

        `0 <= lambda <= 1`

        If `lambda` exists, the light of view will be blocked by that voxel. We check
        first if the distance from the voxel center to the lov is greater than sqrt(3).

    ## Arguments:

        cam_pos (np.ndarray): the position of the camera.

        lov (np.ndarray): the vector from camera pos to target pos.

        occupancy (List[np.ndarray]): the occupancy points of the room.
    """
    CENTER_SHIFT = np.array([0.5, 0.5, 0.5])

    for occ in occupancy:
        # check if the distance from the voxel center to lov is greater than sqrt(3)
        occ_vec = occ - cam_pos
        len_occ_vec = np.linalg.norm(occ_vec)
        cos_theta = np.dot(lov, occ_vec) / (np.linalg.norm(lov) * len_occ_vec)
        sin_theta = math.sin(math.acos(cos_theta))
        dist = len_occ_vec * sin_theta

        if dist > math.sqrt(3):
            continue

        # solve linear programming
        linpro_mat = np.vstack((-lov, lov))
        linpro_const = np.vstack(
            (occ + CENTER_SHIFT - cam_pos - lov, -occ + CENTER_SHIFT + cam_pos + lov)
        )
        res = linprog(
            np.array([1]), A_ub=linpro_mat, b_ub=linpro_const, bounds=((0, 1))
        )
        if not (res.status == 0 and res.success):
            return False

    return True
