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


def action_index2movement(
    action_index: int,
    cam_types: int,
) -> array:
    """
    ## Description:

        Translate action_index to pos, angular and cam_type movements.

    ## Arguments:

        action_index (int)

        cam_types (int)

    ## Returns:

        action (np.ndarray): [6], np.int64,
        [delta_x, delta_y, delta_z, delta_pan_index, delta_tilt_index, cam_type]
    """
    assert isinstance(action_index, int) and isinstance(cam_types, int), (
        f"`action_index` and `cam_types` must both be `int` variables, but got "
        f"{type(action_index)} and {type(cam_types)}."
    )
    assert action_index in range(10 * cam_types), (
        "`action_index` is supposed to be chosen from 0 ~ (10 * `cam_types` - 1), but "
        f"got {action_index}."
    )

    new_cam_type: array = np.array([action_index // 10], dtype=np.int64)

    # movement
    # | index | (x, y, z, p, t) |
    # | ----- | --------------- |
    # |   0   | (-, 0, 0, 0, 0) |
    # |   1   | (+, 0, 0, 0, 0) |
    # |   2   | (0, -, 0, 0, 0) |
    # |   3   | (0, +, 0, 0, 0) |
    # |   4   | (0, 0, -, 0, 0) |
    # |   5   | (0, 0, +, 0, 0) |
    # |   6   | (0, 0, 0, -, 0) |
    # |   7   | (0, 0, 0, +, 0) |
    # |   8   | (0, 0, 0, 0, -) |
    # |   9   | (0, 0, 0, 0, +) |
    spacial_action: array = np.zeros([5], dtype=np.int64)
    spacial_action[action_index % 10 // 2] = -1 if action_index % 10 % 2 == 0 else 1

    return np.concatenate((spacial_action, new_cam_type), axis=0).astype(np.int64)


def apply_movement_on_agent(
    cur_params: array,
    movement: array,
    room_shape: List[int],
    pan_section_num: int,
    tilt_section_num: int,
    pan_range: List[float],
    tilt_range: List[float],
    allow_pan_circular: bool = False,
) -> array:
    """
    ## Description:

        Apply movement to an agent's current pos, angular and cam_type params, and
        return its new params.

    ## Arguments:

        cur_params (np.ndarray): [6], np.float32,
        [cur_x, cur_y, cur_z, cur_pan, cur_tilt, cur_cam_type]

        movement (np.ndarray): [6], np.int64,
        [delta_x, delta_y, delta_z, delta_pan_index, delta_tilt_index, cam_type]

        room_shape (List[int]): [3], W, D, H of the room.

        pan_section_num (int)

        tilt_section_num (int)

        allow_pan_circular (bool): whether allow circular pan angle.

    ## Returns:

        new_params (np.ndarray): [6], np.float32,
        [new_x, new_y, new_z, new_pan, new_tilt, new_cam_type]
    """
    assert len(cur_params) == len(movement) == 6

    # position
    pos_upper_bound = np.array(room_shape) - 1
    pos_lower_bound = np.zeros_like(pos_upper_bound)
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
        new_direction[0] = np.clip(
            cur_params[3] + angular_movement[0] * pan_step,
            pan_lower_bound,
            pan_upper_bound,
        )

    # tilt without 2 polar points
    tilt_upper_bound = np.array([tilt_range[1] - tilt_step])
    tilt_lower_bound = np.array([tilt_range[0] + tilt_step])
    new_direction[1] = np.clip(
        cur_params[4] + angular_movement[1] * tilt_step,
        tilt_lower_bound,
        tilt_upper_bound,
    )

    return np.concatenate(
        (new_pos, new_direction, np.array([cur_params[5]])),
        axis=0,
    ).astype(np.float32)
