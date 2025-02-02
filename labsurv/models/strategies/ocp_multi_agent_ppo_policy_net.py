import torch
import torch.nn.functional as F
from labsurv.builders import STRATEGIES
from torch import Tensor
from torch.nn import Linear, Module


@STRATEGIES.register_module()
class OCPMultiAgentPPOPolicyNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: str,
        hidden_dim: int,
        cam_types: int,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.cam_types = cam_types

        self.hidden_1 = Linear(6, self.hidden_dim, device=self.device)
        self.hidden_2 = Linear(self.hidden_dim, self.hidden_dim, device=self.device)
        self.hidden_3 = Linear(self.hidden_dim, self.hidden_dim, device=self.device)

        self.out = Linear(hidden_dim, 10 * self.cam_types, device=self.device)

    def forward(self, input: Tensor, without_noise: bool) -> Tensor:
        """
        ## Arguments:

            input (Tensor): [B, 6], pos_coord, pan-tilt, and cam_type of the agent.

            without_noise (bool)

        ## Returns:

            dist (Tensor): [B, 10 * CAM_TYPE], action distribution.
        """

        assert input.ndim == 2 and input.shape[1] == 6, "Input should be shaped [B, 6]."
        assert (
            str(input.device).startswith("cuda") and str(self.device).startswith("cuda")
        ) or (
            str(input.device).startswith("cpu") and str(self.device).startswith("cpu")
        ), "Different devices found."
        assert input.dtype == self.FLOAT

        x = F.relu(self.hidden_1(input))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))

        out = self.out(x)

        # if not without_noise:
        #     noise = torch.normal(
        #         0.0, 1.0, size=(10 * self.cam_types,), device=self.device
        #     )
        #     out += noise

        action_dist = F.softmax(out, dim=-1)

        return action_dist  # [B, 10 * CAM_TYPE]
