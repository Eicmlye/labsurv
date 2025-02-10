from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import torch
from numpy import ndarray as array
from torch import Tensor
from torch import pi as PI


def pos_index2coord(
    room_shape: List[int], pos_index: int, device: Optional[torch.cuda.device] = None
) -> Tensor:
    """
    ## Description:

        Change pos_index to pos_coord.
    """
    room: Tensor = torch.ones(room_shape, device=device)
    pos_coord = room.nonzero()[pos_index]

    return pos_coord  # [3]


def array_is_in(pos: array, candidates: array):
    extend_pos = np.tile(pos, [len(candidates), 1])
    return np.any(np.all(extend_pos == candidates, axis=1), axis=0)


def direction_index2pan_tilt(
    pan_section_num: int,
    tilt_section_num: int,
    direction_index: int,
    pan_range: List[float] = [-PI, PI],
    tilt_range: List[float] = [-PI / 2, PI / 2],
    device: Optional[torch.cuda.device] = None,
    ignore_polar: bool = False,
) -> Tensor:
    """
    ## Description:

        Change direction_index to [pan, tilt] format.
    """

    if not ignore_polar:
        if direction_index == 0:
            pan = pan_range[0]
            tilt = tilt_range[0]
        else:
            pan_index = (direction_index - 1) % pan_section_num
            tilt_index = (direction_index - 1) // pan_section_num + 1

            pan = (
                pan_range[0]
                + pan_index * (pan_range[1] - pan_range[0]) / pan_section_num
            )
            tilt = (
                tilt_range[0]
                + tilt_index * (tilt_range[1] - tilt_range[0]) / tilt_section_num
            )
    else:
        raise NotImplementedError()

    return torch.tensor([pan, tilt], device=device)


def observation2input(observation: Tensor) -> Tuple[Tensor, Tensor]:
    """
    ## Description:

        Change [B, 12, W, D, H]-shaped observation to policy net input form.

    ## Returns:

        x (Tensor): [B, 1, W, D, H], 1 for blocked, 2 for visible, 0 for invisible.

        pos_mask (Tensor): [B, 1, W, D, H], 1 for pos that allows installation and yet
        haven't been installed at
    """
    assert isinstance(observation, Tensor)
    assert observation.ndim == 5

    x: Tensor = observation.clone().detach()

    # 1 for blocked, 2 for visible, 0 for invisible
    x = (x[:, 0] + x[:, -1] * 2).unsqueeze(1)  # [B, 1, W, D, H]

    cache_observ: Tensor = observation.clone().detach()
    # pos that allows installation and yet haven't been installed at
    pos_mask: Tensor = torch.logical_xor(
        cache_observ[:, [1]], cache_observ[:, [7]]
    )  # [B, 1, W, D, H]

    return x, pos_mask


def pack_observation2transition(observation: array) -> array:
    """
    ## Description:

        Change [12, W, D, H]-shaped observation to policy net input concatenated form.

    ## Returns:

        output (np.ndarray): [1, 2, W, D, H].
    """
    assert observation.ndim == 4
    obs = torch.tensor(observation).unsqueeze(0)  # [1, 12, W, D, H]

    return torch.cat(observation2input(obs), dim=1).cpu().numpy().copy()


def action2movement(action: array) -> array:
    """
    ## Description:

        Translate action to movement.

    ## Arguments:

        action (np.ndarray)

    ## Returns:

        movement (np.ndarray): [6], np.int64,
        [delta_x, delta_y, delta_z, delta_pan_index, delta_tilt_index, cam_type]
    """

    action_index: int = round(action.nonzero()[0][0])
    movement: array = np.zeros((6,), dtype=np.float32)

    if action_index < 10:
        movement[action_index // 2] = -1 if action_index % 2 == 0 else 1
    else:
        movement[5] = action_index - 10

    return movement


def apply_movement_on_agent(
    cur_params: array,
    action: array,
    room_shape: List[int],
    pan_section_num: int,
    tilt_section_num: int,
    pan_range: List[float],
    tilt_range: List[float],
    allow_pan_circular: bool = False,
    pos_candidate: Optional[array] = None,
) -> array | bool:
    """
    ## Description:

        Apply movement to an agent's current params, and return its new params.

    ## Arguments:

        cur_params (np.ndarray): [PARAM_DIM], np.float32.

        action (np.ndarray): [ACTION_DIM], np.int64.

        room_shape (List[int]): [3], W, D, H of the room.

        pan_section_num (int)

        tilt_section_num (int)

        allow_pan_circular (bool): whether allow circular pan angle.

        pos_candidate (Optional[array]): if not None, this method return False
        immediately if invalid action is executed (run out of `pos_candidate`, or
        angles hit bounds).

    ## Returns:

        new_params (np.ndarray): [PARAM_DIM], np.float32.

        valid_action (bool): only return this when `pos_candidate` is not None.
    """
    movement = action2movement(action)

    # position
    pos_upper_bound = np.array(room_shape) - 1
    pos_lower_bound = np.zeros_like(pos_upper_bound)
    if (
        pos_candidate is not None
        and int(action.nonzero()[0]) < 6
        and not array_is_in(
            (cur_params[:3].copy() + movement[:3].copy()).astype(np.int64),
            pos_candidate.astype(np.int64),
        )
    ):
        return False
    new_pos: array = np.clip(
        cur_params[:3].copy() + movement[:3].copy(), pos_lower_bound, pos_upper_bound
    )

    # direction
    pan_step: float = (pan_range[1] - pan_range[0]) / pan_section_num
    tilt_step: float = (tilt_range[1] - tilt_range[0]) / tilt_section_num
    angular_movement = movement[3:5].copy()

    new_direction: array = np.zeros([2], dtype=np.float32)
    if allow_pan_circular and pan_range[0] == -PI and pan_range[1] == PI:
        # pan is circular
        if np.abs(cur_params[3] - (-PI)) < 1e-5 and angular_movement[0] == -1:
            new_direction[0] = PI - pan_step
        elif np.abs(cur_params[3] + pan_step - PI) < 1e-5 and angular_movement[0] == 1:
            new_direction[0] = -PI
        else:
            new_direction[0] = cur_params[3] + angular_movement[0] * pan_step
    else:
        pan_upper_bound = np.array([pan_range[1] - pan_step])
        pan_lower_bound = np.array([pan_range[0]])
        if pos_candidate is not None and (
            round(((cur_params[3] - pan_lower_bound) / pan_step)[0])
            + angular_movement[0]
            < 0
            or round(((cur_params[3] - pan_lower_bound) / pan_step)[0])
            + angular_movement[0]
            > pan_section_num - 1
        ):
            return False
        new_direction[0] = np.clip(
            cur_params[3] + angular_movement[0] * pan_step,
            pan_lower_bound,
            pan_upper_bound,
        )

    # tilt without 2 polar points
    tilt_upper_bound = np.array([tilt_range[1] - tilt_step])
    tilt_lower_bound = np.array([tilt_range[0]])
    if pos_candidate is not None and (
        round(((cur_params[4] - tilt_lower_bound) / tilt_step)[0]) + angular_movement[1]
        < 1
        or round(((cur_params[4] - tilt_lower_bound) / tilt_step)[0])
        + angular_movement[1]
        > tilt_section_num - 1
    ):
        return False
    new_direction[1] = np.clip(
        cur_params[4] + angular_movement[1] * tilt_step,
        tilt_lower_bound,
        tilt_upper_bound,
    )

    return (
        np.concatenate(
            (
                new_pos,
                new_direction,
                np.eye(len(cur_params) - 5)[cur_params[5:].astype(np.bool_)].reshape(
                    -1
                ),
            ),
            axis=0,
        ).astype(np.float32)
        if pos_candidate is None
        else True
    )


def info_room2critic_input(room, agent_num: int) -> Tuple[array, array]:
    """
    ## Description:

        Generate critic inputs from room info.

    ## Arguments:

        room (SurveillanceRoom): will be deepcopied and will NOT be modified.

        agent_num (int)

    ## Returns:

        cam_params (np.ndarray): [AGENT_NUM, PARAM_DIM], torch.float. Absolute coords
        and relative angles, one-hot cam_type vecs.

        env (np.ndarray): [3, W, D, H], torch.float, `occupancy`,
        `install_permitted`, `vis_redundancy`.
        `vis_redundancy` = vis_count / agent_num, -1 if not in `must_monitor`.
    """
    room = deepcopy(room)

    info_room: array = room.get_info()
    cam_types: int = len(room._CAM_TYPES)

    # env
    occupancy: array = np.expand_dims(info_room[0], axis=0)
    install_permitted: array = np.expand_dims(info_room[1], axis=0)
    must_monitor: array = info_room[2]
    visible_count: array = info_room[11]

    assert agent_num >= np.max(visible_count)
    vis_redundancy: array = visible_count / agent_num
    vis_redundancy[must_monitor == 0] = -1
    vis_redundancy = np.expand_dims(vis_redundancy, axis=0)

    ## agent params
    # [AGENT_NUM, 3]
    cam_pos: array = np.array(info_room[7].nonzero()).transpose()
    # [AGENT_NUM, 1]
    pan: array = info_room[8][info_room[7].nonzero()].reshape(agent_num, 1)
    # [AGENT_NUM, 1]
    tilt: array = info_room[9][info_room[7].nonzero()].reshape(agent_num, 1)
    # [AGENT_NUM , CAM_TYPES]
    cam_type_one_hot: array = np.eye(cam_types)[
        info_room[10][info_room[7].nonzero()].astype(np.int64)
    ]

    # [AGENT_NUM, PARAM_DIM]
    cam_params: array = np.concatenate([cam_pos, pan, tilt, cam_type_one_hot], axis=1)
    # [3, W, D, H]
    env: array = np.concatenate([occupancy, install_permitted, vis_redundancy], axis=0)

    return cam_params, env


def info_room2actor_input(
    room, agent_num: int, cur_cam_params: array
) -> Tuple[array, array, array]:
    """
    ## Description:

        Generate actor inputs from room info according to current camera position.

    ## Arguments:

        room (SurveillanceRoom): will be deepcopied and will NOT be modified.

        agent_num (int)

        cur_cam_params (np.ndarray): [PARAM_DIM].

    ## Returns:

        self_and_neigh_params (np.ndarray): [AGENT_NUM(NEIGH), PARAM_DIM], torch.float.
        Absolute params of agents in `neigh` including current agent itself. Remaining
        rows will be padded with 0's.

        self_mask (np.ndarray): [AGENT_NUM(NEIGH)], torch.bool. 1 for current agent, 0
        otherwise. Remaining rows will be padded with `False`'s.

        neigh (np.ndarray): [3, 2L+1, 2L+1, 2L+1], torch.float, where `L` is the
        farther dof of the camera. `occupancy`, `install_permitted`,
        `vis_redundancy`. `vis_redundancy` = vis_count / agent_num, -1 if not
        in `must_monitor`.
    """
    room = deepcopy(room)

    info_room: array = room.get_info()
    cam_types: int = len(room._CAM_TYPES)

    ## full env
    occupancy: array = np.expand_dims(info_room[0], axis=0)
    install_permitted: array = np.expand_dims(info_room[1], axis=0)
    must_monitor: array = info_room[2]
    visible_count: array = info_room[11]

    assert agent_num >= np.max(visible_count)
    vis_redundancy: array = visible_count / agent_num
    vis_redundancy[must_monitor == 0] = -1
    vis_redundancy = np.expand_dims(vis_redundancy, axis=0)
    # [3, W, D, H]
    env: array = np.concatenate([occupancy, install_permitted, vis_redundancy], axis=0)

    ## local env
    cur_cam_type = round(np.array(cur_cam_params[5:].nonzero()).reshape(-1)[0])
    cur_cam_intrinsics = room.get_cam_intrinsics(cur_cam_type)
    if "dof" not in cur_cam_intrinsics.keys():
        raise NotImplementedError("Only support explicit params.")
    L: int = min(max(round(cur_cam_intrinsics["dof"][0] / room.voxel_length), 5), 10)
    W, D, H = env[0].shape

    x_min = round(max(0, cur_cam_params[0] - L))
    x_max = round(min(cur_cam_params[0] + L, W - 1))
    y_min = round(max(0, cur_cam_params[1] - L))
    y_max = round(min(cur_cam_params[1] + L, D - 1))
    z_min = round(max(0, cur_cam_params[2] - L))
    z_max = round(min(cur_cam_params[2] + L, H - 1))

    real_neigh: array = env[:, x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
    padded_neigh: array = np.concatenate(
        [
            [
                np.zeros([2 * L + 1, 2 * L + 1, 2 * L + 1]),
                np.zeros([2 * L + 1, 2 * L + 1, 2 * L + 1]),
                -1 * np.ones([2 * L + 1, 2 * L + 1, 2 * L + 1]),
            ]
        ],
        axis=1,
    )

    x_negative_bound_dis = round(cur_cam_params[0] - x_min)
    x_positive_bound_dis = round(x_max - cur_cam_params[0])
    y_negative_bound_dis = round(cur_cam_params[1] - y_min)
    y_positive_bound_dis = round(y_max - cur_cam_params[1])
    z_negative_bound_dis = round(cur_cam_params[2] - z_min)
    z_positive_bound_dis = round(z_max - cur_cam_params[2])
    padded_neigh[
        :,
        L - x_negative_bound_dis : L + x_positive_bound_dis + 1,
        L - y_negative_bound_dis : L + y_positive_bound_dis + 1,
        L - z_negative_bound_dis : L + z_positive_bound_dis + 1,
    ] = real_neigh

    ## params
    neigh_extrin: array = np.zeros_like(info_room[7])
    neigh_extrin[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1] = info_room[
        7
    ][x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
    # [AGENT_NUM(NEIGH), 3]
    self_and_neigh_cam_pos: array = np.array(neigh_extrin.nonzero()).transpose()
    # [AGENT_NUM(NEIGH), 1]
    self_and_neigh_pan: array = info_room[8][neigh_extrin.nonzero()].reshape(-1, 1)
    # [AGENT_NUM(NEIGH), 1]
    self_and_neigh_tilt: array = info_room[9][neigh_extrin.nonzero()].reshape(-1, 1)
    # [AGENT_NUM(NEIGH), CAM_TYPES]
    self_and_neigh_cam_type_one_hot: array = np.eye(cam_types)[
        info_room[10][neigh_extrin.nonzero()].astype(np.int64)
    ]
    # [AGENT_NUM(NEIGH), PARAM_DIM]
    self_and_neigh_params: array = np.concatenate(
        [
            self_and_neigh_cam_pos,
            self_and_neigh_pan,
            self_and_neigh_tilt,
            self_and_neigh_cam_type_one_hot,
        ],
        axis=1,
    )

    param_dim = self_and_neigh_params.shape[1]
    padded_self_and_neigh_params = np.zeros((agent_num, param_dim), dtype=np.float32)
    padded_self_and_neigh_params[: len(self_and_neigh_params)] = self_and_neigh_params
    self_mask: array = np.all(
        np.round(self_and_neigh_cam_pos - cur_cam_params[:3]).astype(np.int64) == 0,
        axis=1,
    )
    padded_self_mask = np.zeros((agent_num,), dtype=np.bool_)
    padded_self_mask[: len(self_mask)] = self_mask

    return padded_self_and_neigh_params, padded_self_mask, padded_neigh


def generate_action_mask(
    room,
    cam_params: array,
    pan_section_num: int,
    tilt_section_num: int,
    pan_range: List[float],
    tilt_range: List[float],
) -> array:
    """
    ## Description:

        Generate mask for invalid actions.

    ## Arguments:

        room (SurveillanceRoom)

        cam_params (np.ndarray): [PARAM_DIM] or [B, PARAM_DIM]

    ## Returns:

        valid_actions (np.ndarray): [ACTION_DIM] or [B, ACTION_DIM]
    """
    assert cam_params.ndim in [1, 2]
    if cam_params.ndim == 1:
        return _generate_action_mask(
            room,
            cam_params,
            pan_section_num,
            tilt_section_num,
            pan_range,
            tilt_range,
        )
    else:
        batched_masks = []
        for cam_param in cam_params:
            batched_masks.append(
                _generate_action_mask(
                    room,
                    cam_param,
                    pan_section_num,
                    tilt_section_num,
                    pan_range,
                    tilt_range,
                )
            )
        return np.array(batched_masks)


def _generate_action_mask(
    room,
    cam_params: array,
    pan_section_num: int,
    tilt_section_num: int,
    pan_range: List[float],
    tilt_range: List[float],
) -> array:
    """
    ## Description:

        Generate mask for invalid actions.

    ## Arguments:

        room (SurveillanceRoom)

        cam_params (np.ndarray): [PARAM_DIM]

    ## Returns:

        valid_actions (np.ndarray): [ACTION_DIM]
    """
    room = deepcopy(room)

    pos_candidates = (
        (
            torch.logical_xor(
                room.install_permitted,
                room.cam_extrinsics[:, :, :, 0],
            )
            .nonzero()
            .type(torch.int64)
        )
        .cpu()
        .numpy()
        .copy()
    )
    cur_cam_type = int(cam_params[5:].nonzero()[0])

    actions = np.eye(5 + len(cam_params))
    valid_actions = np.ones((5 + len(cam_params),))
    for index, action in enumerate(actions):
        if index >= 10:
            is_valid = cur_cam_type != int(action[10:].nonzero()[0])
        else:
            is_valid = apply_movement_on_agent(
                cam_params,
                action,
                room.shape,
                pan_section_num,
                tilt_section_num,
                pan_range,
                tilt_range,
                pos_candidate=pos_candidates,
            )
        if not is_valid:
            valid_actions[index] = 0

    return valid_actions  # [ACTION_DIM]
