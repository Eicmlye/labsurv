from typing import List, Optional, Tuple

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
    device: Optional[torch.cuda.device] = None,
) -> Tensor:
    """
    ## Description:

        Change direction_index to [pan, tilt] format.
    """
    if direction_index == 0:
        pan = -PI
        tilt = -PI / 2
    else:
        pan_index = (direction_index - 1) % pan_section_num
        tilt_index = (direction_index - 1) // pan_section_num + 1

        pan = -PI + pan_index * 2 * PI / pan_section_num
        tilt = -PI / 2 + tilt_index * PI / 2 / tilt_section_num

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
