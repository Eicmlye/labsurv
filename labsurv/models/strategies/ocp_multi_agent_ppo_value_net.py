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
class OCPMultiAgentPPOValueNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: str,
        neck_hidden_dim: int,
        adaptive_pooling_dim: int,
        neck_layers: int,
        neck_out_use_num: int,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.neck_out_use_num = neck_out_use_num

        self.neck = []
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
                        (
                            "bn",
                            BatchNorm3d(
                                neck_hidden_dim // 2**layer, device=self.device
                            ),
                        ),
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

        neck_output_cat_dim = (
            neck_hidden_dim * (2**self.neck_out_use_num - 1) // 2 ** (neck_layers - 1)
        )

        self.adaptive_pooling = AdaptiveMaxPool3d(
            (adaptive_pooling_dim, adaptive_pooling_dim, adaptive_pooling_dim)
        )

        self.linear = Linear(
            in_features=(neck_output_cat_dim * adaptive_pooling_dim**3),
            out_features=1,
            dtype=self.FLOAT,
            device=self.device,
        )
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 5, (
            "Input `x` should be shaped [B, 1, W, D, H], " f"but got {x.shape}."
        )
        assert (
            str(x.device).startswith("cuda") and str(self.device).startswith("cuda")
        ) or (
            str(x.device).startswith("cpu") and str(self.device).startswith("cpu")
        ), "Different devices found."
        assert x.dtype == self.FLOAT

        # 1 for blocked, 2 for visible, 0 for else
        x = (x[:, 0] + x[:, -1] * 2).unsqueeze(1)  # [B, 1, W, H ,D]

        neck_output = [x]
        for neck_layer in self.neck:
            neck_output.append(neck_layer(neck_output[-1]))

        head_input = torch.cat(neck_output[-self.neck_out_use_num :], dim=1)

        output: Tensor = self.adaptive_pooling(head_input)

        output = self.linear(output.flatten(start_dim=1))

        return output  # [B, 1]
