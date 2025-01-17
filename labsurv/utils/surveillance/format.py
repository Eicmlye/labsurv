from typing import Optional

import torch
from torch import Tensor
from torch import pi as PI


def pos_index2coord(
    room_shape: torch.Size, pos_index: int, device: Optional[torch.cuda.device] = None
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
    if direction_index == 0:
        pan = -PI
        tilt = -PI / 2
    else:
        pan_index = (direction_index - 1) % pan_section_num
        tilt_index = (direction_index - 1) // pan_section_num + 1

        pan = -PI + pan_index * 2 * PI / pan_section_num
        tilt = -PI / 2 + tilt_index * PI / tilt_section_num

    return torch.tensor([pan, tilt], device=device)
