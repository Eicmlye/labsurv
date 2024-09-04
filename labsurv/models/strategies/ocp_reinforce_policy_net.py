from typing import OrderedDict, Tuple

import torch
import torch.nn.functional as F
from labsurv.builders import STRATEGIES
from torch import Tensor
from torch.nn import (
    AdaptiveMaxPool3d,
    BatchNorm3d,
    Conv3d,
    Linear,
    MaxPool3d,
    Module,
    ReLU,
    Sequential,
)


@STRATEGIES.register_module()
class OCPREINFORCEPolicyNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: torch.cuda.device,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        params_dim: int,
    ):
        super().__init__()
        self.device = device

        neck_layers = 2
        self.neck = Sequential()
        for layer in range(neck_layers):
            cur_layer = Sequential(
                OrderedDict(
                    [
                        (  # conv
                            "conv",
                            Conv3d(
                                in_channels=(
                                    state_dim
                                    if layer == 0
                                    else hidden_dim // 2 ** (layer - 1)
                                ),
                                out_channels=hidden_dim // 2**layer,
                                kernel_size=3,
                                padding=1,
                                dtype=self.FLOAT,
                                device=self.device,
                            ),
                        ),
                        ("bn", BatchNorm3d(hidden_dim // 2**layer)),
                        ("relu", ReLU()),
                        (  # maxpool
                            "maxpool",
                            MaxPool3d(
                                kernel_size=2,
                                stride=2,
                                padding=0,
                            ),
                        ),
                    ]
                )
            )

            # NOTE(eric): using kaiming init is important. w/o that, the linear layer
            # barely learns anything.
            torch.nn.init.kaiming_normal_(
                cur_layer.conv.weight, mode="fan_out", nonlinearity="relu"
            )
            torch.nn.init.constant_(cur_layer.conv.bias, 0)
            self.neck.append(cur_layer)

        self.action_head = Sequential(
            OrderedDict(
                [
                    (  # conv
                        "conv",
                        Conv3d(
                            in_channels=hidden_dim // 2 ** (neck_layers - 1),
                            out_channels=hidden_dim // 2**neck_layers,
                            kernel_size=3,
                            padding=1,
                            dtype=self.FLOAT,
                            device=self.device,
                        ),
                    ),
                    ("bn", BatchNorm3d(hidden_dim // 2**neck_layers)),
                    ("relu", ReLU()),
                    (  # adaptive_maxpool
                        "adaptive_maxpool",
                        AdaptiveMaxPool3d(output_size=(4, 4, 4)),
                    ),
                ]
            )
        )
        torch.nn.init.kaiming_normal_(
            self.action_head.conv.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.action_head.conv.bias, 0)
        self.action_out = Linear(
            hidden_dim // 2**neck_layers * 4**3,
            action_dim,
            dtype=self.FLOAT,
            device=self.device,
        )

        self.params_head = Sequential(
            OrderedDict(
                [
                    (  # conv
                        "conv",
                        Conv3d(
                            in_channels=hidden_dim // 2 ** (neck_layers - 1),
                            out_channels=hidden_dim // 2**neck_layers,
                            kernel_size=3,
                            padding=1,
                            dtype=self.FLOAT,
                            device=self.device,
                        ),
                    ),
                    ("bn", BatchNorm3d(hidden_dim // 2**neck_layers)),
                    ("relu", ReLU()),
                    (  # adaptive_maxpool
                        "adaptive_maxpool",
                        AdaptiveMaxPool3d(output_size=(4, 4, 4)),
                    ),
                ]
            )
        )
        torch.nn.init.kaiming_normal_(
            self.params_head.conv.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.params_head.conv.bias, 0)
        self.params_out = Linear(
            hidden_dim // 2**neck_layers * 4**3,
            2 * params_dim,
            dtype=self.FLOAT,
            device=self.device,
        )

    def forward(self, observation: Tensor) -> Tuple[Tensor, Tensor]:
        assert observation.ndim == 5, "Input should be shaped [B, C, W, D, H]."
        assert (
            str(observation.device).startswith("cuda")
            and str(self.device).startswith("cuda")
        ) or (
            str(observation.device).startswith("cpu")
            and str(self.device).startswith("cpu")
        ), "Different devices found."
        assert observation.dtype == self.FLOAT

        x = self.neck(observation)

        action = torch.flatten(self.action_head(x))
        action = self.action_out(action)

        # del_cam and adjust_cam are only available when any cam is installed
        if torch.sign(observation[0, 7].sum()) == 0:
            action[[0, 3]] = F.softmax(action[[0, 3]], dim=0)
            action[[1, 2]] = 0
        else:
            action = F.softmax(action, dim=0)

        params = torch.flatten(self.params_head(x))
        params = self.params_out(params)  # [mu, sigma] * 4

        params[[1, 3, 5, 7]] = torch.abs(params[[1, 3, 5, 7]])  # positive sigmas
        # params[[0, 2, 4, 6]] = torch.atan(params[[0, 2, 4, 6]]) * 2 / torch.pi  # mu in (-1, 1)
        # params[[0, 2, 4, 6]] = torch.sin(params[[0, 2, 4, 6]])  # mu in (-1, 1)
        params[[0, 2, 4, 6]] = (
            torch.sigmoid(params[[0, 2, 4, 6]]) * 2 - 1
        )  # mu in (-1, 1)

        return action, params
