from typing import List, OrderedDict, Tuple

import torch
import torch.nn.functional as F
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


class OCPPPOPositionPolicyNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: torch.cuda.device,
        in_channels: int,
    ):
        super().__init__()
        self.device = device

        self.out = Conv3d(  # [B, 1, W, D, H]
            in_channels=in_channels,
            out_channels=1,
            kernel_size=3,
            padding=1,
            dtype=self.FLOAT,
            device=self.device,
        )
        torch.nn.init.kaiming_normal_(
            self.out.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.out.bias, 0)

    def forward(self, x: Tensor, pos_mask: Tensor):
        x = F.sigmoid(self.out(x)) * pos_mask.type(self.INT)
        x[x == 0] = float("-inf")

        return x


class OCPPPOProvidedPositionParamsPolicyNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: torch.cuda.device,
        features_in_channels: int,
        out_features: int,
        adaptive_pooling_dim: int,
    ):
        super().__init__()
        self.device = device

        self.conv = Conv3d(  # [B, 1, W, D, H]
            in_channels=features_in_channels + 1,
            out_channels=(features_in_channels + 1) // 2,
            kernel_size=3,
            padding=1,
            dtype=self.FLOAT,
            device=self.device,
        )
        torch.nn.init.kaiming_normal_(
            self.conv.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.conv.bias, 0)

        self.adaptive_pooling = AdaptiveMaxPool3d(
            (
                adaptive_pooling_dim,
                adaptive_pooling_dim,
                adaptive_pooling_dim,
            )
        )

        self.out = Linear(
            in_features=(features_in_channels + 1) // 2 * adaptive_pooling_dim**3,
            out_features=out_features,
            dtype=self.FLOAT,
            device=self.device,
        )
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, pos_dist: Tensor):
        # [B, features_in_channels + 1, W, D, H]
        x = torch.cat((x, pos_dist), dim=1)

        x = self.conv(x)

        return F.sigmoid(self.out(self.adaptive_pooling(x).flatten(start_dim=1)))


@STRATEGIES.register_module()
class OCPPPOPolicyNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: str,
        hidden_dim: int,
        action_dim: int,
        cam_types: int,
        neck_layers: int,
        neck_out_use_num: int,
        pan_section_num: int = 360,
        tilt_section_num: int = 180,
        adaptive_pooling_dim: int = 10,
    ):
        assert neck_out_use_num <= neck_layers

        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        self.action_dim = action_dim
        self.pan_section_num = pan_section_num
        self.tilt_section_num = tilt_section_num
        self.neck_out_use_num = neck_out_use_num
        self.adaptive_pooling_dim = adaptive_pooling_dim

        self.neck: List[Module] = []
        for layer in range(neck_layers):
            cur_layer = Sequential(
                OrderedDict(
                    [
                        (  # conv
                            "conv",
                            Conv3d(
                                in_channels=(
                                    2 if layer == 0 else hidden_dim // 2 ** (layer - 1)
                                ),
                                out_channels=hidden_dim // 2**layer,
                                kernel_size=3,
                                padding=1,
                                dtype=self.FLOAT,
                                device=self.device,
                            ),
                        ),
                        ("bn", BatchNorm3d(hidden_dim // 2**layer, device=self.device)),
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
            hidden_dim * (2**self.neck_out_use_num - 1) // 2 ** (neck_layers - 1)
        )

        self.pos_head = OCPPPOPositionPolicyNet(
            device=self.device,
            in_channels=neck_output_cat_dim,
        )

        self.params_head = OCPPPOProvidedPositionParamsPolicyNet(
            device=self.device,
            features_in_channels=neck_output_cat_dim,
            out_features=(self.pan_section_num * (self.tilt_section_num - 1) + 1)
            * self.cam_types,
            adaptive_pooling_dim=self.adaptive_pooling_dim,
        )

    def forward(self, input: Tensor, pos_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        ## Arguments:

            input (Tensor): [B, 1, W, D, H], 1 for blocked, 2 for visible, 0 for invisible.

            pos_mask (Tensor): [B, 1, W, D, H], pos that allows installation and yet
                haven't been installed at.

        ## Returns:

            pos_dist (Tensor): [B, 1, W, D, H].

            direction_index_dist (Tensor): [B, DIRECTION].

            cam_type_dist (Tensor): [B, CAM_TYPE].
        """

        assert input.ndim == 5, "Input should be shaped [B, 1, W, D, H]."
        assert (
            str(input.device).startswith("cuda") and str(self.device).startswith("cuda")
        ) or (
            str(input.device).startswith("cpu") and str(self.device).startswith("cpu")
        ), "Different devices found."
        assert input.dtype == self.FLOAT
        batch_size, _, width, depth, height = input.shape
        x: Tensor = torch.cat((input, pos_mask), dim=1)

        neck_output = [x]
        for neck_layer in self.neck:
            neck_output.append(neck_layer(neck_output[-1]))

        head_input = torch.cat(neck_output[-self.neck_out_use_num :], dim=1)

        pos_output: Tensor = self.pos_head(head_input, pos_mask)  # [B, 1, W, D, H]
        pos_dist = _batchwise_softmax(pos_output)  # [B, W * D * H]

        param_output: Tensor = self.params_head(
            head_input,
            pos_dist.view(batch_size, -1, width, depth, height),
        )  # [B, DIRECTION * CAM_TYPE]
        param_dist = _batchwise_softmax(param_output)  # [B, DIRECTION * CAM_TYPE]

        return pos_dist, param_dist


def _batchwise_softmax(input: Tensor) -> Tensor:
    batch_size = input.shape[0]
    return F.softmax(input.view(batch_size, -1), dim=1)  # [B, ...]
