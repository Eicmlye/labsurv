from typing import OrderedDict

import torch
from labsurv.builders import STRATEGIES
from torch import Tensor
from torch.nn import (
    AdaptiveMaxPool3d,
    BatchNorm3d,
    Conv3d,
    Linear,
    Module,
    ReLU,
    Sequential,
)


@STRATEGIES.register_module()
class OCPDDPGCleanValueNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: str,
        neck_hidden_dim: int,
        adaptive_pooling_dim: int,
        neck_layers: int = 3,
    ):
        super().__init__()
        self.device = torch.device(device)

        self.neck = Sequential()
        for layer in range(neck_layers):
            cur_layer = Sequential(
                OrderedDict(
                    [
                        (  # conv
                            "conv",
                            Conv3d(
                                in_channels=(
                                    1
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

        self.linear = Linear(
            in_features=(
                neck_hidden_dim // 2 ** (neck_layers - 1) * adaptive_pooling_dim**3 + 7
            ),
            out_features=1,
            dtype=self.FLOAT,
            device=self.device,
        )

    def forward(self, x: Tensor, action_with_params: Tensor) -> Tensor:
        assert x.ndim == 5, (
            "Input `x` should be shaped [B, 1, W, D, H], " f"but got {x.shape}."
        )
        assert action_with_params.ndim == 2 and action_with_params.shape[1] == 7, (
            "Input `action` should be shaped [B, 7], "
            f"but got {action_with_params.shape}."
        )
        assert (
            str(x.device).startswith("cuda") and str(self.device).startswith("cuda")
        ) or (
            str(x.device).startswith("cpu") and str(self.device).startswith("cpu")
        ), "Different devices found."
        assert x.dtype == self.FLOAT

        # 1 for blocked, 2 for visible, 0 for else
        x = (x[:, 0] + x[:, -1] * 2).unsqueeze(1)  # [B, 1, W, H ,D]

        x = self.neck(x)
        x = self.adaptive_pooling(x)
        x = torch.cat((x.flatten(start_dim=1), action_with_params), dim=1)

        x = self.linear(x)

        return x  # B * 1
