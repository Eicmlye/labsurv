from typing import Dict, List

import torch
from mmcv.utils import ProgressBar
from torch import Tensor


def normalize_coords_list(tensor: Tensor) -> Tensor:
    """
    ## Description:

        Normalize a list of tensors.

    ## Arguments:

        tensor (Tensor): [N, 3].
    """
    return tensor / torch.linalg.vector_norm(tensor, dim=1).view(-1, 1)


def compute_single_cam_visibility(
    cam_pos: Tensor,
    direction: Tensor,
    intrinsic: Dict[str, float | List[float]],
    occupancy: Tensor,
    target: Tensor,
    voxel_length: float,
) -> Tensor:
    """
    ## Description:

        Compute the visible point mask of a single camera.

    ## Arguments:

        cam_pos (Tensor): [3], torch.int64, the position of the camera.

        direction (Tensor): [2], torch.float16, the `pan` and `tilt` of the camera.

        intrinsic (Dict[str, float | List[float]]): camera intrinsics.

        occupancy (Tensor): [W, D, H], torch.int64, occupancy mask.

        target (Tensor): [W, D, H, 4], torch.float16,
        [need_monitor, h_res_req_min, h_res_req_max, v_res_req_min, v_res_req_max].

        voxel_length (float): length of sides of voxels in meters.

    ## Returns:

        vis_mask_3d (Tensor): [W, D, H], torch.int64, the visibility mask of the camera.
    """
    if (
        "clip_shape" in intrinsic.keys()
        and "focal_length" in intrinsic.keys()
        and "resolution" in intrinsic.keys()
    ):
        return _compute_single_cam_visibility_with_raw_params(
            cam_pos,
            direction,
            intrinsic,
            occupancy,
            target,
            voxel_length,
        )
    else:
        return _compute_single_cam_visibility_with_explicit_params(
            cam_pos,
            direction,
            intrinsic,
            occupancy,
            target,
            voxel_length,
        )


def _compute_single_cam_visibility_with_raw_params(
    cam_pos: Tensor,
    direction: Tensor,
    intrinsic: Dict[str, float | List[float]],
    occupancy: Tensor,
    target: Tensor,
    voxel_length: float,
) -> Tensor:
    """
    ## Arguments:

        cam_pos (Tensor): [3], torch.int64, the position of the camera.

        direction (Tensor): [2], torch.float16, the `pan` and `tilt` of the camera.

        intrinsic (Dict[str, float | List[float]]): camera intrinsics.

        occupancy (Tensor): [W, D, H], torch.int64, occupancy mask.

        target (Tensor): [W, D, H, 4], torch.float16,
        [need_monitor, h_res_req_min, h_res_req_max, v_res_req_min, v_res_req_max].

        voxel_length (float): length of sides of voxels in meters.

    ## Returns:

        vis_mask_3d (Tensor): [W, D, H], torch.int64, the visibility mask of the camera.
    """
    device = occupancy.device
    INT = torch.int64
    FLOAT = torch.float

    assert cam_pos.dtype == occupancy.dtype == INT
    assert cam_pos.device == direction.device == target.device == device

    pan = direction[0]
    tilt = direction[1]
    clip_shape = torch.tensor(
        intrinsic["clip_shape"], dtype=FLOAT, device=device
    )  # horizontal, vertical
    f = torch.tensor([intrinsic["focal_length"]], dtype=FLOAT, device=device)
    resolution = torch.tensor(
        intrinsic["resolution"], dtype=FLOAT, device=device
    )  # horizontal, vertical

    rot_mat = torch.tensor(
        [
            [torch.cos(pan), -torch.sin(pan), 0],
            [torch.sin(pan), torch.cos(pan), 0],
            [0, 0, 1],
        ],
        dtype=FLOAT,
        device=device,
    ) @ torch.tensor(
        [
            [torch.cos(tilt), 0, -torch.sin(tilt)],
            [0, 1, 0],
            [torch.sin(tilt), 0, torch.cos(tilt)],
        ],
        dtype=FLOAT,
        device=device,
    )

    # tangent values of half of angle of view, [horizontal, vertical]
    tan_aov_half = clip_shape / (2 * f)

    # change all lov from world coord to cam coord
    target_mask = target[:, :, :, 0]
    all_coords = target_mask.nonzero()  # N * 3
    lov = (all_coords - cam_pos).type(FLOAT)  # N * 3
    lov_in_cam_coord = (rot_mat.permute(1, 0) @ lov.permute(1, 0)).permute(
        1, 0
    )  # N * 3

    # check aov
    aov_coord_mask = check_aov(lov_in_cam_coord, tan_aov_half)

    # check dof
    # N * 4, [h_res_req_min, h_res_req_max, v_res_req_min, v_res_req_max]
    all_extrinsics = target[all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]][:, 1:]
    # N * 4, [h_far, h_near, v_far, v_near]
    view_plane_shapes = resolution.repeat_interleave(2) / all_extrinsics
    candidate_dof = f * view_plane_shapes / clip_shape.repeat_interleave(2)
    # N * 2, [d_far, d_near]
    dof = torch.cat(
        (
            torch.min(candidate_dof[:, [0, 2]], dim=1).values.view(-1, 1),
            torch.max(candidate_dof[:, [1, 3]], dim=1).values.view(-1, 1),
        ),
        dim=1,
    )
    dof_coord_mask = (lov_in_cam_coord[:, 0] * voxel_length <= dof[:, 0]) & (
        lov_in_cam_coord[:, 0] * voxel_length >= dof[:, 1]
    )

    # check obstacles
    obstacle_mask = check_obstacle_dda(cam_pos, all_coords, occupancy)

    visibility_mask = aov_coord_mask & dof_coord_mask & obstacle_mask
    visible_coords = all_coords[visibility_mask]

    vis_mask_3d = torch.zeros_like(occupancy, device=device)
    vis_mask_3d[visible_coords[:, 0], visible_coords[:, 1], visible_coords[:, 2]] = 1

    return vis_mask_3d


def _compute_single_cam_visibility_with_explicit_params(
    cam_pos: Tensor,
    direction: Tensor,
    intrinsic: Dict[str, float | List[float]],
    occupancy: Tensor,
    target: Tensor,
    voxel_length: float,
) -> Tensor:
    """
    ## CAUTION:

        Explicit params of intrinsics suggest that the resolution requirements are depending
        on the cameras, not the target points. In this case, target resolution requirements
        are ignored.

    ## Arguments:

        cam_pos (Tensor): [3], torch.int64, the position of the camera.

        direction (Tensor): [2], torch.float16, the `pan` and `tilt` of the camera.

        intrinsic (Dict[str, float | List[float]]): camera intrinsics.

        occupancy (Tensor): [W, D, H], torch.int64, occupancy mask.

        target (Tensor): [W, D, H, 4], torch.float16,
        [need_monitor, h_res_req_min, h_res_req_max, v_res_req_min, v_res_req_max].

        voxel_length (float): length of sides of voxels in meters.

    ## Returns:

        vis_mask_3d (Tensor): [W, D, H], torch.int64, the visibility mask of the camera.
    """
    device = occupancy.device
    INT = torch.int64
    FLOAT = torch.float

    assert cam_pos.dtype == occupancy.dtype == INT
    assert cam_pos.device == direction.device == target.device == device

    pan = direction[0]
    tilt = direction[1]
    aov = torch.tensor(
        intrinsic["aov"], dtype=FLOAT, device=device
    )  # horizontal, vertical
    dof = torch.tensor(intrinsic["dof"], dtype=FLOAT, device=device)  # far, near

    assert aov[0] > 0 and aov[0] < torch.pi
    assert aov[1] > 0 and aov[1] < torch.pi
    assert dof[0] > dof[1]

    rot_mat: Tensor = torch.tensor(
        [
            [torch.cos(pan), -torch.sin(pan), 0],
            [torch.sin(pan), torch.cos(pan), 0],
            [0, 0, 1],
        ],
        dtype=FLOAT,
        device=device,
    ) @ torch.tensor(
        [
            [torch.cos(tilt), 0, -torch.sin(tilt)],
            [0, 1, 0],
            [torch.sin(tilt), 0, torch.cos(tilt)],
        ],
        dtype=FLOAT,
        device=device,
    )

    # tangent values of half of angle of view, [horizontal, vertical]
    tan_aov_half = torch.tan(aov / 2)

    # change all lov from world coord to cam coord
    target_mask = target[:, :, :, 0]
    all_coords = target_mask.nonzero()  # N * 3
    lov = (all_coords - cam_pos).type(FLOAT)  # N * 3
    lov_in_cam_coord = (rot_mat.permute(1, 0) @ lov.permute(1, 0)).permute(
        1, 0
    )  # N * 3

    # check aov
    aov_coord_mask = check_aov(lov_in_cam_coord, tan_aov_half)

    # check dof
    # N * 2, [d_far, d_near]
    all_dof = dof.repeat([lov.shape[0], 1])
    dof_coord_mask = (lov_in_cam_coord[:, 0] * voxel_length <= all_dof[:, 0]) & (
        lov_in_cam_coord[:, 0] * voxel_length >= all_dof[:, 1]
    )

    # check obstacles
    obstacle_mask = check_obstacle_dda(cam_pos, all_coords, occupancy)

    visibility_mask = aov_coord_mask & dof_coord_mask & obstacle_mask
    visible_coords = all_coords[visibility_mask]

    vis_mask_3d = torch.zeros_like(occupancy, device=device)
    vis_mask_3d[visible_coords[:, 0], visible_coords[:, 1], visible_coords[:, 2]] = 1

    return vis_mask_3d


def check_aov(lov_in_cam_coord: Tensor, tan_aov_half: Tensor):
    """
    ## Description:

        Generate aov visibility mask.

    ## Arguments:

        lov_in_cam_coord (Tensor): [N, 3], the coords of all lov's w.r.t. the camera.

        tan_aov_half (Tensor): [horizontal, vertical], tangent values of half of angle
        of view.
    """
    all_coords_cam_coord_unit = normalize_coords_list(lov_in_cam_coord)  # N * 3
    tan_horizontal = torch.abs(
        all_coords_cam_coord_unit[:, 1] / all_coords_cam_coord_unit[:, 0]
    )
    tan_vertical = torch.abs(
        all_coords_cam_coord_unit[:, 2]
        / torch.linalg.vector_norm(all_coords_cam_coord_unit[:, :2], dim=1)
    )
    aov_coord_mask = (
        (all_coords_cam_coord_unit[:, 0] > 0)  # zero/neg means behind the camera.
        & (tan_horizontal <= tan_aov_half[0])
        & (tan_vertical <= tan_aov_half[1])
    )

    return aov_coord_mask


def check_obstacle_dda(cam_pos: Tensor, targets: Tensor, occupancy: Tensor) -> Tensor:
    """
    Generated by DeepSeek R1
    Checked and modified by Eric

    ## Description:

        Compute visibility from `cam_pos` to each target in voxel space using 3D DDA
        algorithm.

    ## Arguments:

        cam_pos (Tensor): [3], camera position coord.

        targets (Tensor): [N, 3], target point coords.

        occupancy (Tensor): [W, D, H].

    ## Returns:

        Tensor: [N], 1 for visible, 0 otherwise.
    """

    device = targets.device
    target_num = targets.shape[0]

    # distance along each axis
    delta = targets - cam_pos.unsqueeze(0)
    dx = delta[:, 0]
    dy = delta[:, 1]
    dz = delta[:, 2]

    # step forward direction
    step_x = torch.sign(dx).to(torch.int32)
    step_y = torch.sign(dy).to(torch.int32)
    step_z = torch.sign(dz).to(torch.int32)

    # absolute distance
    dx_abs = dx.abs().to(torch.int32)
    dy_abs = dy.abs().to(torch.int32)
    dz_abs = dz.abs().to(torch.int32)

    # init current position
    current_pos = cam_pos.unsqueeze(0).repeat(target_num, 1).to(torch.int32)

    # vismask
    visible = torch.ones(target_num, dtype=torch.bool, device=device)
    # if a ray detected occupancy, deactivate it
    active = (current_pos != targets).any(dim=1)

    # init parameterized distances
    t_max_x = torch.full((target_num,), float('inf'), device=device)
    t_max_y = torch.full((target_num,), float('inf'), device=device)
    t_max_z = torch.full((target_num,), float('inf'), device=device)

    mask_x = dx_abs != 0
    t_max_x[mask_x] = (
        current_pos[mask_x, 0].float()
        + torch.abs(step_x[mask_x]).float()
        - cam_pos[0].float()
    ) / dx_abs[mask_x].float()

    mask_y = dy_abs != 0
    t_max_y[mask_y] = (
        current_pos[mask_y, 1].float()
        + torch.abs(step_y[mask_y]).float()
        - cam_pos[1].float()
    ) / dy_abs[mask_y].float()

    mask_z = dz_abs != 0
    t_max_z[mask_z] = (
        current_pos[mask_z, 2].float()
        + torch.abs(step_z[mask_z]).float()
        - cam_pos[2].float()
    ) / dz_abs[mask_z].float()

    # distance for every step forward
    t_delta_x = torch.ones(target_num, device=device) / dx_abs.float().clamp(min=1e-9)
    t_delta_x[dx_abs == 0] = float('inf')

    t_delta_y = torch.ones(target_num, device=device) / dy_abs.float().clamp(min=1e-9)
    t_delta_y[dy_abs == 0] = float('inf')

    t_delta_z = torch.ones(target_num, device=device) / dz_abs.float().clamp(min=1e-9)
    t_delta_z[dz_abs == 0] = float('inf')

    print("\nDetecting obstacles...")
    prog_bar = ProgressBar(torch.abs(delta).sum(dim=1).max())
    while torch.any(active):
        # find minimal t_max and corresponding axis
        t_max = torch.stack([t_max_x, t_max_y, t_max_z], dim=1)
        min_t, min_axis = torch.min(t_max, dim=1)

        # generate mask for coord update for each axis
        mask_x = (min_axis == 0) & active
        mask_y = (min_axis == 1) & active
        mask_z = (min_axis == 2) & active

        # update coord
        current_pos[mask_x, 0] += step_x[mask_x]
        current_pos[mask_y, 1] += step_y[mask_y]
        current_pos[mask_z, 2] += step_z[mask_z]

        # update t_max
        t_max_x[mask_x] += t_delta_x[mask_x]
        t_max_y[mask_y] += t_delta_y[mask_y]
        t_max_z[mask_z] += t_delta_z[mask_z]

        # check if hit obstacle
        x = current_pos[:, 0].long()
        y = current_pos[:, 1].long()
        z = current_pos[:, 2].long()
        voxel_values = occupancy[x, y, z]

        hit = (voxel_values == 1) & active
        visible[hit] = False
        active[hit] = False

        # check if reached target points
        reached = (current_pos == targets).all(dim=1) & active
        active[reached] = False

        prog_bar.update()
    print("\r\033[K\033[1A\033[K\033[1A", end="")

    return visible.to(torch.bool)
