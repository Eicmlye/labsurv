from typing import List, Optional

import torch
import torch.nn.functional as F
from labsurv.builders import STRATEGIES
from torch import Tensor
from torch.nn import Linear, Module, Sequential


@STRATEGIES.register_module()
class SimplePolicyNet(Module):
    def __init__(
        self,
        device: torch.cuda.device,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        extra_params: Optional[List[List[float]]] = None,
    ):
        super().__init__()
        self.device = device

        self.conv1 = Linear(state_dim, hidden_dim).to(self.device)

        # [is_discrete, lbound, ubound(, include_lbound, include_ubound)]
        self.param_intervals = extra_params
        if self.param_intervals is not None:
            self.conv2 = Linear(hidden_dim, hidden_dim // 2).to(self.device)
            self.conv_action = Linear(hidden_dim // 2, action_dim).to(self.device)
            self.conv_param = []

            for interval in self.param_intervals:
                seq = Sequential().to(self.device)

                if interval[0]:  # discrete, classification head
                    seq.append(
                        Linear(hidden_dim // 2, interval[2] - interval[1] + 1).to(
                            self.device
                        )
                    )
                else:  # continuous, regression head
                    seq.append(Linear(hidden_dim // 2, 1).to(self.device))

                self.conv_param.append(seq)
        else:
            self.conv_action = Linear(hidden_dim, action_dim).to(self.device)

    def forward(self, observation):
        x = F.relu(self.conv1(observation))
        EPSILON = 1e-7

        if self.param_intervals is not None:
            x = F.relu(self.conv2(x))
            actions = F.softmax(self.conv_action(x), dim=1)

            params = []
            for index, interval in enumerate(self.param_intervals):
                if interval[0]:  # discrete, classification head
                    param = F.softmax(self.conv_param[index](x), dim=1).argmax().item()
                else:  # continuous, regression head
                    # clip output values to intervals
                    param = torch.max(
                        torch.min(
                            self.conv_param[index](x),
                            Tensor(
                                [interval[2] - (EPSILON if not interval[4] else 0)]
                            ).to(x.device),
                        ),
                        Tensor([interval[1] + (EPSILON if not interval[3] else 0)]).to(
                            x.device
                        ),
                    ).item()

                params.append(param)

            params = torch.Tensor(params).to(self.device)

            return actions, params
        else:
            actions = F.softmax(self.conv_action(x), dim=0)
            return actions
