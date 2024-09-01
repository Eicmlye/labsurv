import torch
from torch import Tensor


def shift(points: Tensor, displacement: Tensor, int_output: bool = False):
    """
    ## Arguments:

        points (Tensor): [3] or [N, 3], torch.float16 or torch.int64.

        displaycement (Tensor): [3], torch.float16 or torch.int64.

    ## Returns:

        result_points (Tensor): [3] or [N, 3], NEW tensor(s) shifted, torch.int64.
    """
    device = points.device
    INT = torch.int64
    FLOAT = torch.float16
    assert displacement.device == device, "Different devices found."

    result_points = points.type(FLOAT)
    displacement = displacement.type(FLOAT)
    if not displacement.equal(torch.tensor([0, 0, 0], dtype=FLOAT, device=device)):
        full_disp = displacement.repeat(points.shape[0], 1)
        result_points += full_disp

    if int_output:
        result_points = result_points.type(INT)

    return result_points


def rotate(points: Tensor, rot_mat: Tensor):
    pass
