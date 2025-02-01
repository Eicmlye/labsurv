from typing import Dict, List, Tuple

import torch
from mmcv.utils import ProgressBar
from numpy import ndarray as array
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
    lov_indices: List[int] = None,
    lov_check_list: List[List[int]] = None,
) -> Tensor:
    """
    ## Description:

        Compute the visible point coords of a single camera.

    ## Arguments:

        cam_pos (Tensor): [3], torch.int64, the position of the camera.

        direction (Tensor): [2], torch.float16, the `pan` and `tilt` of the camera.

        intrinsic (Dict[str, float | List[float]]): camera intrinsics.

        occupancy (Tensor): [W, D, H], torch.int64, occupancy mask.

        target (Tensor): [W, D, H, 4], torch.float16,
        [need_monitor, h_res_req_min, h_res_req_max, v_res_req_min, v_res_req_max].

        voxel_length (float): length of sides of voxels in meters.

        lov_indices (Optional[List[int]]): auxiliary list for obstacle check
        speedup. Generate this list by `if_need_obstacle_check()`.

        lov_check_list (Optional[List[List[int]]): auxiliary list for obstacle
        check speedup. Generate this list by `if_need_obstacle_check()`.

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
            lov_indices,
            lov_check_list,
        )
    else:
        return _compute_single_cam_visibility_with_explicit_params(
            cam_pos,
            direction,
            intrinsic,
            occupancy,
            target,
            voxel_length,
            lov_indices,
            lov_check_list,
        )


def _compute_single_cam_visibility_with_raw_params(
    cam_pos: Tensor,
    direction: Tensor,
    intrinsic: Dict[str, float | List[float]],
    occupancy: Tensor,
    target: Tensor,
    voxel_length: float,
    lov_indices: List[int] = None,
    lov_check_list: List[List[int]] = None,
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

        lov_indices (Optional[List[int]]): auxiliary list for obstacle check
        speedup. Generate this list by `if_need_obstacle_check()`.

        lov_check_list (Optional[List[List[int]]): auxiliary list for obstacle
        check speedup. Generate this list by `if_need_obstacle_check()`.

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
    all_coords_cam_coord = (rot_mat.permute(1, 0) @ lov.permute(1, 0)).permute(
        1, 0
    )  # N * 3

    # check aov
    all_coords_cam_coord_unit = normalize_coords_list(all_coords_cam_coord)  # N * 3
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
    dof_coord_mask = (all_coords_cam_coord[:, 0] * voxel_length <= dof[:, 0]) & (
        all_coords_cam_coord[:, 0] * voxel_length >= dof[:, 1]
    )
    # check obstacles
    obstacle_mask = check_obstacle(cam_pos, lov, occupancy, lov_indices, lov_check_list)

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
    lov_indices: List[int] = None,
    lov_check_list: List[List[int]] = None,
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

        vis_mask (Tensor): [W, D, H], torch.int64, the visibility mask of the camera.
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
    tan_aov_half = torch.tan(aov / 2)

    # change all lov from world coord to cam coord
    target_mask = target[:, :, :, 0]
    all_coords = target_mask.nonzero()  # N * 3
    lov = (all_coords - cam_pos).type(FLOAT)  # N * 3
    all_coords_cam_coord = (rot_mat.permute(1, 0) @ lov.permute(1, 0)).permute(
        1, 0
    )  # N * 3

    # check aov
    all_coords_cam_coord_unit = normalize_coords_list(all_coords_cam_coord)  # N * 3
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

    # check dof
    # N * 2, [d_far, d_near]
    all_dof = dof.repeat([lov.shape[0], 1])
    dof_coord_mask = (all_coords_cam_coord[:, 0] * voxel_length <= all_dof[:, 0]) & (
        all_coords_cam_coord[:, 0] * voxel_length >= all_dof[:, 1]
    )

    # check obstacles
    # obstacle_mask = check_obstacle(cam_pos, lov, occupancy, lov_indices, lov_check_list)
    obstacle_mask = check_obstacle_bresenham(cam_pos, all_coords, occupancy)

    visibility_mask = aov_coord_mask & dof_coord_mask & obstacle_mask
    # DEBUG(eric)
    # visibility_mask = obstacle_mask
    visible_coords = all_coords[visibility_mask]

    result = torch.zeros_like(occupancy, device=device)
    result[visible_coords[:, 0], visible_coords[:, 1], visible_coords[:, 2]] = 1

    return result


def check_obstacle(
    cam_pos: Tensor,
    lov: Tensor,
    occupancy: Tensor,
    lov_indices: List[int] = None,
    lov_check_list: List[List[int]] = None,
):
    r"""
    ## Description:

        Check if the light of view is block by any obstacles. This criterion computes
        if the lov passes any occupancy voxel. The lov is continuous.

        This is equivalant to 6 linear programming problems, checking if the
        intersections of the lov segment and the 6 surfaces of the voxel are between
        the other surfaces of dimensions. Specifically, let camera be P, target be Q
        and the center of voxel be T, then solving \lambda from

        x_P + \lambda_{x-}\delta_x = x_T - 0.5, where \delta_x = PQ_x = lov_x

        => \lambda_{x-} = \dfrac{x_T - 0.5 - x_P}{\delta_x}

        one finds the intersection I coords by

        x_I = x_P + \lambda_{x-}\delta_x = x_T - 0.5
        y_I = y_P + \lambda_{x-}\delta_y in (y_T - 0.5, y_T + 0.5) => invisible
        z_I = z_P + \lambda_{x-}\delta_z in (z_T - 0.5, z_T + 0.5) => invisible

        The above is one linear programming problem, and solve 5 more with x_T - 0.5
        and for y_T and z_T.

    ## Arguments:

        cam_pos (Tensor): [3], torch.int64, the position of the camera.

        lov (Tensor): [N, 3], torch.int64, the vector from camera pos to target pos.

        occupancy (Tensor): [W, D, H], torch.int64, occupancy mask.

    ## Returns:

        obstacle_mask (Tensor): [N], torch.bool, visibility mask respect to
        obstacles.
    """
    device = occupancy.device
    assert cam_pos.device == lov.device == device

    occ = occupancy.nonzero().type(torch.float)  # N * 3
    voxel_lower_bounds = occ - 0.5  # N * 3
    voxel_upper_bounds = occ + 0.5  # N * 3
    voxel_bounds = torch.cat((voxel_lower_bounds, voxel_upper_bounds), dim=1)  # N * 6

    obstacle_mask = torch.ones([len(lov)], dtype=torch.bool, device=device)

    if lov_indices is None and lov_check_list is None:
        lov_indices, lov_check_list = _if_need_obstacle_check(cam_pos, lov, occ)

    print("\nChecking if camera's light of view passes any obstacles...")
    prog_bar = ProgressBar(len(lov_check_list))
    for index, target in enumerate(lov_check_list):
        target_vec = torch.tensor(target, dtype=torch.int64, device=device)
        lov_index = lov_indices[index]
        lambdas = (voxel_bounds - torch.cat((cam_pos, cam_pos))) / torch.cat(
            (target_vec, target_vec)
        )  # N * 6, [x-, y-, z-, x+, y+, z+]
        intersections = cam_pos.repeat(6) + lambdas.repeat_interleave(
            3, 1
        ) * target_vec.repeat(
            6
        )  # N * 18
        # [
        #   x_{I, x-}, y_{I, x-}, z_{I, x-},
        #   x_{I, y-}, y_{I, y-}, z_{I, y-},
        #   x_{I, z-}, y_{I, z-}, z_{I, z-},
        #   x_{I, x+}, y_{I, x+}, z_{I, x+},
        #   x_{I, y+}, y_{I, y+}, z_{I, y+},
        #   x_{I, z+}, y_{I, z+}, z_{I, z+},
        # ]

        equal_bounds_mask = torch.tensor(
            [False, True, True, True, False, True, True, True, False], device=device
        ).repeat(occ.shape[0], 2)
        intersection_mask = (
            (intersections >= voxel_lower_bounds.repeat(1, 6))
            & (intersections <= voxel_upper_bounds.repeat(1, 6))
            & equal_bounds_mask
        )  # N * 18
        # CAUTION: For the above mask, if use > and < without equal_bounds_mask, there
        # is chance that lov passes the edge of the voxel, in which case the target
        # points will be mistakenly checked as visible.

        obstacle_mask[lov_index] = (
            (intersection_mask.view(-1, 6, 3).sum(dim=2) == 2)
            & (lambdas >= 0)
            & (lambdas <= 1)
        ).sum() == 0

        prog_bar.update()
    print("\r\033[K\033[1A\033[K\033[1A\033[K\033[1A")

    return obstacle_mask


def if_need_obstacle_check(
    cam_pos: array,
    must_monitor: Tensor,
    occupancy: Tensor,
    step: int = 500,
) -> Tuple[List, List]:
    """
    ## Description:

        Check if the occ points are too far away from the lov segment.

    ## Arguments:

        cam_pos (array): [3], np.int64, the position of the camera.

        must_monitor (Tensor): [W, D, H, 4]

        occupancy (Tensor): [W, D, H, 1]

        step (int): batch size for computation, in case of OOM for huge rooms.

    ## Returns:

        lov_indices (List): [LAMBDA], the indices of the lov that need further check in
        the original lov tensor.

        lov_check_list (List): [LAMBDA, 3], coords of the lov that need further check.
    """

    device = must_monitor.device

    cam_pos_vec = torch.tensor(cam_pos, dtype=torch.int64, device=device)  # [3]

    target_mask = must_monitor.clone().detach()[:, :, :, 0]
    all_coords = target_mask.nonzero()  # N * 3
    # the vector from camera pos to target pos
    lov = (all_coords - cam_pos_vec).type(torch.float32)  # N * 3

    occ = occupancy.clone().detach().nonzero().type(torch.float)  # M * 3

    return _if_need_obstacle_check(cam_pos_vec, lov, occ, step)


def _if_need_obstacle_check(
    cam_pos: Tensor,
    lov: Tensor,
    occ: Tensor,
    step: int = 500,
) -> Tuple[List, List]:
    """
    ## Description:

        Check if the occ points are too far away from the lov segment.

    ## Arguments:

        cam_pos (Tensor): [3], torch.int64, the position of the camera.

        lov (Tensor): [N, 3], torch.int64, the vector from camera pos to target pos.

        occ (Tensor): [M, 3], torch.int64, the coords of the occupancy points.

        step (int): batch size for computation, in case of OOM for huge rooms.

    ## Returns:

        lov_indices (List): [LAMBDA], the indices of the lov that need further check in
        the original lov tensor.

        lov_check_list (List): [LAMBDA, 3], coords of the lov that need further check.
    """

    lov_indices = []
    lov_check_list = []

    print("\nBuilding speed up indices...")
    prog_bar = ProgressBar(len(lov) // step + (1 if (len(lov) % step > 0) else 0))
    for index in range(len(lov) // step):
        lov_section = lov[step * index : step * (index + 1)]

        # ignore check if occ is too far away from the lov line
        lov_needs_check_dist_mask = _check_dist_occ2lov(lov_section, cam_pos, occ)
        # Ignore check if occ is not `between` the cam_pos and target.
        #
        # --------occ------------
        # cam--------------target    NEED CHECK
        #
        # occ-------------
        # -------------cam-------------target    NO CHECK
        #
        # If occ is inside the vertex voxels, check is still necessary.
        # lov_needs_check_between_mask = _check_occ_between_lov(
        #     lov_section, cam_pos, occ, dist_occ2lov
        # )
        lov_needs_check_mask = (
            lov_needs_check_dist_mask  # & lov_needs_check_between_mask
        )

        lov_needs_check_indices = (lov_needs_check_mask > 0).nonzero().view(
            -1
        ) + index * step
        lov_indices += lov_needs_check_indices.tolist()
        lov_check_list += lov[lov_needs_check_indices].tolist()

        prog_bar.update()

    if len(lov) % step > 0:
        lov_section = lov[len(lov) // step * step :]

        lov_needs_check_dist_mask = _check_dist_occ2lov(lov_section, cam_pos, occ)
        # lov_needs_check_between_mask = _check_occ_between_lov(
        #     lov_section, cam_pos, occ, dist_occ2lov
        # )
        lov_needs_check_mask = (
            lov_needs_check_dist_mask  # & lov_needs_check_between_mask
        )

        lov_needs_check_indices = (lov_needs_check_mask > 0).nonzero().view(-1) + len(
            lov
        ) // step * step
        lov_indices += lov_needs_check_indices.tolist()
        lov_check_list += lov[lov_needs_check_indices].tolist()

        prog_bar.update()

    print("\r\033[K\033[1A\033[K\033[1A", end="")

    # DEBUG(eric)
    # lov_indices = [i for i in range(len(lov))]
    # lov_check_list = lov.tolist()

    return lov_indices, lov_check_list


def _check_dist_occ2lov(lov_section: Tensor, cam_pos: Tensor, occ: Tensor) -> Tensor:
    """
    ## Arguments:

        lov_section (Tensor): [K, 3], torch.int64, the vector from camera pos to target
        pos.

        cam_pos (Tensor): [3], torch.int64, the position of the camera.

        occ (Tensor): [M, 3], torch.int64, the coords of the occupancy points.
    """

    cross_products = torch.sum(
        torch.cross(
            lov_section.unsqueeze(1),
            (cam_pos.repeat([len(occ), 1]) - occ).unsqueeze(0),
            dim=2,
        )
        ** 2,
        dim=2,
    )  # K * M
    dist_occ2lov_squared_sum = cross_products / torch.sum(
        lov_section**2, dim=1
    ).unsqueeze(1).repeat(
        [1, len(occ)]
    )  # K * M

    # check occ only if distance <= sqrt(3)
    lov_needs_check_mask = (dist_occ2lov_squared_sum < 3).sum(dim=1) > 1

    return lov_needs_check_mask  # , torch.sqrt(dist_occ2lov_squared_sum)  # [K], [K, M]


def _check_occ_between_lov(
    lov_section: Tensor, cam_pos: Tensor, occ: Tensor, dist_occ2lov: Tensor
) -> Tensor:
    """
    ## Arguments:

        lov_section (Tensor): [K, 3], torch.int64, the vector from camera pos to target
        pos.

        cam_pos (Tensor): [3], torch.int64, the position of the camera.

        occ (Tensor): [M, 3], torch.int64, the coords of the occupancy points.

        dist_occ2lov (Tensor): [K * M], torch.float32, the distance from occ to lov
        line.
    """

    targets = lov_section + cam_pos  # [K, 3]

    targets_expanded = targets.repeat([1, len(occ)]).view(-1, len(occ), 3)  # [K, M, 3]
    occ_expanded = occ.repeat([len(targets), 1]).view(len(targets), -1, 3)  # [K, M, 3]
    occ_to_target = targets_expanded - occ_expanded  # [K, M, 3]

    # For 3D meshgrids, `near` is equivalent to being the same voxel.
    near_vertices_mask = (occ_to_target == 0).sum(dim=2) == 3  # [K, M]

    target_side_squared = (occ_to_target**2).sum(dim=2) - dist_occ2lov**2  # [K, M]
    target_side_inner_mask = (
        target_side_squared
        - (lov_section.repeat([1, len(occ)]).view(-1, len(occ), 3) ** 2).sum(dim=2)
    ) <= 0  # [K, M]

    cam_expanded = cam_pos.repeat([len(occ) * len(lov_section)]).view(
        len(targets), len(occ), 3
    )  # [K, M, 3]
    occ_to_cam = cam_expanded - occ_expanded  # [K, M, 3]

    cam_side_squared = (occ_to_cam**2).sum(dim=2) - dist_occ2lov**2  # [K, M]
    cam_side_inner_mask = (
        cam_side_squared
        - (lov_section.repeat([1, len(occ)]).view(-1, len(occ), 3) ** 2).sum(dim=2)
    ) <= 0  # [K, M]

    lov_needs_check_between_mask = (
        near_vertices_mask | (target_side_inner_mask & cam_side_inner_mask)
    ).sum(
        dim=1
    ) > 0  # [K]

    return lov_needs_check_between_mask


def check_obstacle_bresenham(
    start: Tensor, targets: Tensor, grid: Tensor
) -> torch.Tensor:
    """
    Generated by DeepSeek R1
    Checked and modified by Eric

    Compute visibility from start to each target in voxel space using 3D DDA algorithm.

    Args:
        start (torch.Tensor): Shape [3], 起始点坐标 (整数)
        targets (torch.Tensor): Shape [N, 3], N个目标点坐标 (整数)
        grid (torch.Tensor): Shape [W, D, H], 体素网格, 1表示遮挡物, 0表示空

    Returns:
        torch.Tensor: Shape [N], 1表示可见, 0表示被遮挡
    """

    device = targets.device
    N = targets.shape[0]

    # 计算各轴的差量
    delta = targets - start.unsqueeze(0)
    dx = delta[:, 0]
    dy = delta[:, 1]
    dz = delta[:, 2]

    # 步进方向（+1或-1）
    step_x = torch.sign(dx).to(torch.int32)
    step_y = torch.sign(dy).to(torch.int32)
    step_z = torch.sign(dz).to(torch.int32)

    # 绝对差量
    dx_abs = dx.abs().to(torch.int32)
    dy_abs = dy.abs().to(torch.int32)
    dz_abs = dz.abs().to(torch.int32)

    # 初始化当前位置
    current_pos = start.unsqueeze(0).repeat(N, 1).to(torch.int32)

    # 可见性标记
    visible = torch.ones(N, dtype=torch.bool, device=device)
    active = (current_pos != targets).any(dim=1)  # 需要处理的射线

    # 初始化t_max和t_delta (使用浮点数避免整数溢出)
    t_max_x = torch.full((N,), float('inf'), device=device)
    t_max_y = torch.full((N,), float('inf'), device=device)
    t_max_z = torch.full((N,), float('inf'), device=device)

    # 计算初始t_max（仅当差量不为零时）
    mask = dx_abs != 0
    t_max_x[mask] = (
        current_pos[mask, 0].float()
        + torch.abs(step_x[mask]).float()
        - start[0].float()
    ) / dx_abs[mask].float()

    mask = dy_abs != 0
    t_max_y[mask] = (
        current_pos[mask, 1].float()
        + torch.abs(step_y[mask]).float()
        - start[1].float()
    ) / dy_abs[mask].float()

    mask = dz_abs != 0
    t_max_z[mask] = (
        current_pos[mask, 2].float()
        + torch.abs(step_z[mask]).float()
        - start[2].float()
    ) / dz_abs[mask].float()

    # 计算t_delta（差量为零时设为inf）
    t_delta_x = torch.ones(N, device=device) / dx_abs.float().clamp(min=1e-9)
    t_delta_x[dx_abs == 0] = float('inf')

    t_delta_y = torch.ones(N, device=device) / dy_abs.float().clamp(min=1e-9)
    t_delta_y[dy_abs == 0] = float('inf')

    t_delta_z = torch.ones(N, device=device) / dz_abs.float().clamp(min=1e-9)
    t_delta_z[dz_abs == 0] = float('inf')

    while torch.any(active):
        # 计算当前最小t_max和对应的轴
        t_max = torch.stack([t_max_x, t_max_y, t_max_z], dim=1)
        min_t, min_axis = torch.min(t_max, dim=1)

        # 生成更新掩码
        mask_x = (min_axis == 0) & active
        mask_y = (min_axis == 1) & active
        mask_z = (min_axis == 2) & active

        # 更新当前坐标
        current_pos[mask_x, 0] += step_x[mask_x]
        current_pos[mask_y, 1] += step_y[mask_y]
        current_pos[mask_z, 2] += step_z[mask_z]

        # 更新t_max
        t_max_x[mask_x] += t_delta_x[mask_x]
        t_max_y[mask_y] += t_delta_y[mask_y]
        t_max_z[mask_z] += t_delta_z[mask_z]

        # 检查是否命中遮挡物
        x = current_pos[:, 0].long()
        y = current_pos[:, 1].long()
        z = current_pos[:, 2].long()
        voxel_values = grid[x, y, z]

        hit = (voxel_values == 1) & active
        visible[hit] = False
        active[hit] = False

        # 检查是否到达目标点
        reached = (current_pos == targets).all(dim=1) & active
        active[reached] = False

    return visible.to(torch.bool)
