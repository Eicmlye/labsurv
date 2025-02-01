import torch
import torch.nn.functional as F
from labsurv.builders import STRATEGIES
from torch import Tensor
from torch.nn import Linear, Module


@STRATEGIES.register_module()
class OCPMultiAgentPPOValueNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: str,
        hidden_dim: int,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim

        self.hidden_1 = Linear(6, self.hidden_dim, device=self.device)
        self.hidden_2 = Linear(self.hidden_dim, self.hidden_dim, device=self.device)
        self.hidden_3 = Linear(self.hidden_dim, self.hidden_dim, device=self.device)

        self.out = Linear(self.hidden_dim, 1, device=self.device)

    def forward(self, input: Tensor) -> Tensor:
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

        x = self.out(x)

        return x  # [B, 1]
