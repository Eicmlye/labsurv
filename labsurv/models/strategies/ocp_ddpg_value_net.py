from typing import OrderedDict

import torch
from labsurv.builders import STRATEGIES
from torch import Tensor
from torch.nn import (
    AdaptiveMaxPool3d,
    BatchNorm1d,
    BatchNorm3d,
    Conv3d,
    Linear,
    Module,
    ReLU,
    Sequential,
)


@STRATEGIES.register_module()
class OCPDDPGValueNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: torch.cuda.device,
        state_dim: int,
        neck_hidden_dim: int,
        adaptive_pooling_dim: int,
        hidden_dim: int,
        neck_layers: int = 3,
        linear_layers: int = 3,
    ):
        super().__init__()
        self.device = device

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
                                    else neck_hidden_dim // 2 ** (layer - 1)
                                ),
                                out_channels=neck_hidden_dim // 2**layer,
                                kernel_size=3,
                                padding=1,
                                dtype=self.FLOAT,
                                device=self.device,
                            ),
                        ),
                        ("bn", BatchNorm3d(neck_hidden_dim // 2**layer)),
                        ("relu", ReLU(inplace=True)),
                    ]
                )
            )

            # NOTE(eric): using kaiming init is necessary. w/o that, the conv layer
            # barely learns anything.
            torch.nn.init.kaiming_normal_(
                cur_layer.conv.weight, mode="fan_out", nonlinearity="relu"
            )
            torch.nn.init.constant_(cur_layer.conv.bias, 0)

            self.neck.append(cur_layer)

        self.adaptive_pooling = AdaptiveMaxPool3d(
            (adaptive_pooling_dim, adaptive_pooling_dim, adaptive_pooling_dim)
        )

        self.linear = Sequential()
        for layer in range(linear_layers):
            cur_layer = Sequential(
                OrderedDict(
                    [
                        (  # linear
                            "linear",
                            Linear(
                                in_features=(
                                    neck_hidden_dim
                                    // 2 ** (neck_layers - 1)
                                    * adaptive_pooling_dim**3
                                    + 7
                                    if layer == 0
                                    else hidden_dim // 2 ** (layer - 1)
                                ),
                                out_features=(hidden_dim // 2**layer),
                                dtype=self.FLOAT,
                                device=self.device,
                            ),
                        ),
                        ("bn", BatchNorm1d(hidden_dim // 2**layer)),
                        ("relu", ReLU(inplace=True)),
                    ]
                )
            )

            # NOTE(eric): using kaiming init is necessary. w/o that, the linear layer
            # barely learns anything.
            torch.nn.init.kaiming_normal_(
                cur_layer.linear.weight, mode="fan_out", nonlinearity="relu"
            )
            torch.nn.init.constant_(cur_layer.linear.bias, 0)

            self.linear.append(cur_layer)

        self.out = Linear(
            hidden_dim // 2 ** (linear_layers - 1),
            1,
            dtype=self.FLOAT,
            device=self.device,
        )

    def forward(self, observation: Tensor, action_with_params: Tensor) -> Tensor:
        assert observation.ndim == 5, (
            "`observation` should be shaped [B, C, W, D, H], "
            f"but got {observation.shape}."
        )
        assert action_with_params.ndim == 2 and action_with_params.shape[1] == 7, (
            "`action` should be shaped [B, 7], " f"but got {action_with_params.shape}."
        )
        assert (
            str(observation.device).startswith("cuda")
            and str(self.device).startswith("cuda")
        ) or (
            str(observation.device).startswith("cpu")
            and str(self.device).startswith("cpu")
        ), "Different devices found."
        assert observation.dtype == self.FLOAT

        x: Tensor = observation.clone().detach()
        x[:, [8, 9, 10]] *= x[:, [7]]

        x = self.neck(x)
        x = self.adaptive_pooling(x)
        x = torch.cat((x.flatten(start_dim=1), action_with_params), dim=1)

        x = self.linear(x)

        x = self.out(x)

        return x  # B * 1
