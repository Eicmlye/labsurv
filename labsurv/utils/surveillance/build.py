from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

COLOR_MAP = dict(
    red=np.array([255, 0, 0]),
    green=np.array([0, 255, 0]),
    blue=np.array([0, 0, 255]),
    yellow=np.array([255, 255, 0]),
    grey=np.array([128, 128, 128]),
    white=np.array([255, 255, 255]),
)


def build_block(shape: List[int], device: Optional[torch.cuda.device] = None) -> Tensor:
    """
    ## Arguments:

        shape (List[int]): [3], the shape of the block.

        device (Optional[torch.cuda.device]): the device where all tensors are placed.

    ## Returns:

        points (Tensor): [N, 3], torch.int64, all the point coordinates in the block
        given.
    """
    if len(shape) != 3 or not (
        isinstance(shape[0], int)
        and isinstance(shape[1], int)
        and isinstance(shape[2], int)
    ):
        raise ValueError(
            "A block should be in 3d shape with all side lengths integers."
        )

    # resolution of the block
    points = torch.ones(shape, device=device).nonzero().type(torch.int64)

    return points
