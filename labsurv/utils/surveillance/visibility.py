from typing import Dict, List

import torch
from mmcv.utils import ProgressBar
from torch import Tensor


def normalize_coords_list(tensor: Tensor) -> Tensor:
    """
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
    FLOAT = torch.float16

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
    all_coords_cam_coord_unit = normalize_coords_list(all_coords_cam_coord)  # N * 3

    # check aov
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
    obstacle_mask = check_obstacle(cam_pos, lov, occupancy)

    visibility_mask = aov_coord_mask & dof_coord_mask & obstacle_mask
    visible_coords = all_coords[visibility_mask]

    result = torch.zeros_like(occupancy, device=device)
    result[visible_coords[:, 0], visible_coords[:, 1], visible_coords[:, 2]] = 1

    return result


def check_obstacle(cam_pos: Tensor, lov: Tensor, occupancy: Tensor):
    r"""
    ## Description:

        Check if the light of view is block by any obstacles.

        This is 6 linear programming problem, checking if the intersections of the lov
        segment and the 6 surfaces of the voxel are between the other surfaces of
        dimensions. Specifically, let camera be P, target be Q and the center of voxel
        be T, then solving \lambda from

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

    obstacle_mask = torch.zeros([len(lov)], dtype=torch.bool, device=device)

    print("Checking if camera's light of view passes any obstacles...")
    prog_bar = ProgressBar(len(lov))
    for index, target in enumerate(lov):
        lambdas = (voxel_bounds - torch.cat((cam_pos, cam_pos))) / torch.cat(
            (target, target)
        )  # N * 6, [x-, y-, z-, x+, y+, z+]
        intersections = cam_pos.repeat(6) + lambdas.repeat_interleave(
            3, 1
        ) * target.repeat(
            6
        )  # N * 18
        # [
        #   x_{I,x-}, y_{I, x-}, z_{I, x-},
        #   x_{I,y-}, y_{I, y-}, z_{I, y-},
        #   x_{I,z-}, y_{I, z-}, z_{I, z-},
        #   x_{I,x+}, y_{I, x+}, z_{I, x+},
        #   x_{I,y+}, y_{I, y+}, z_{I, y+},
        #   x_{I,z+}, y_{I, z+}, z_{I, z+},
        # ]

        intersection_mask = (intersections > voxel_lower_bounds.repeat(1, 6)) & (
            intersections < voxel_upper_bounds.repeat(1, 6)
        )  # N * 18
        obstacle_mask[index] = (
            (intersection_mask.view(-1, 6, 3).sum(dim=2) == 2)
            & (lambdas >= 0)
            & (lambdas <= 1)
        ).sum() == 0

        prog_bar.update()
    print("\r\033[K\033[1A\033[K\033[1A")

    return obstacle_mask
