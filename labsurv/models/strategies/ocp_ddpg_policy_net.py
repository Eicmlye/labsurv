from typing import OrderedDict

import torch
from labsurv.builders import STRATEGIES
from torch import Tensor
from torch.nn import BatchNorm3d, Conv3d, Module, ReLU, Sequential, Sigmoid


@STRATEGIES.register_module()
class OCPDDPGPolicyNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: torch.cuda.device,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        cam_types: int,
        neck_layers: int,
    ):
        super().__init__()
        self.device = device
        self.cam_types = cam_types

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

        self.action_head = Sequential(  # B * ACTION * W * D * H
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
                    ("relu", ReLU(inplace=True)),
                    (  # conv_out
                        "conv_out",
                        Conv3d(
                            in_channels=hidden_dim // 2**neck_layers,
                            out_channels=action_dim,
                            kernel_size=3,
                            padding=1,
                            dtype=self.FLOAT,
                            device=self.device,
                        ),
                    ),
                    ("sigmoid", Sigmoid()),
                ]
            )
        )
        torch.nn.init.kaiming_normal_(
            self.action_head.conv.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.action_head.conv.bias, 0)
        torch.nn.init.kaiming_normal_(
            self.action_head.conv_out.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.action_head.conv_out.bias, 0)

        self.pos_head = Sequential(  # B * 1 * W * D * H
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
                    ("relu", ReLU(inplace=True)),
                    (  # conv_out
                        "conv_out",
                        Conv3d(
                            in_channels=hidden_dim // 2**neck_layers,
                            out_channels=1,
                            kernel_size=3,
                            padding=1,
                            dtype=self.FLOAT,
                            device=self.device,
                        ),
                    ),
                    ("sigmoid", Sigmoid()),
                ]
            )
        )
        torch.nn.init.kaiming_normal_(
            self.pos_head.conv.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.pos_head.conv.bias, 0)
        torch.nn.init.kaiming_normal_(
            self.pos_head.conv_out.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.pos_head.conv_out.bias, 0)

        self.direction_head = Sequential(  # B * 2 * W * D * H
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
                    ("relu", ReLU(inplace=True)),
                    (  # conv_out
                        "conv_out",
                        Conv3d(
                            in_channels=hidden_dim // 2**neck_layers,
                            out_channels=2,
                            kernel_size=3,
                            padding=1,
                            dtype=self.FLOAT,
                            device=self.device,
                        ),
                    ),
                ]
            )
        )
        torch.nn.init.kaiming_normal_(
            self.direction_head.conv.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.direction_head.conv.bias, 0)
        torch.nn.init.kaiming_normal_(
            self.direction_head.conv_out.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.direction_head.conv_out.bias, 0)

        self.cam_type_head = Sequential(  # B * CAM_TYPE * W * D * H
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
                    ("relu", ReLU(inplace=True)),
                    (  # conv_out
                        "conv_out",
                        Conv3d(
                            in_channels=hidden_dim // 2**neck_layers,
                            out_channels=self.cam_types,
                            kernel_size=3,
                            padding=1,
                            dtype=self.FLOAT,
                            device=self.device,
                        ),
                    ),
                    ("sigmoid", Sigmoid()),
                ]
            )
        )
        torch.nn.init.kaiming_normal_(
            self.cam_type_head.conv.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.cam_type_head.conv.bias, 0)
        torch.nn.init.kaiming_normal_(
            self.cam_type_head.conv_out.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.cam_type_head.conv_out.bias, 0)

    def forward(self, observation: Tensor) -> Tensor:
        """
        ## Arguments:

            observation (Tensor): B * C * W * D * H.

        ## Returns:

            action_with_params (Tensor): B * 7.
        """

        assert observation.ndim == 5, "Input should be shaped [B, C, W, D, H]."
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

        x: Tensor = self.neck(observation)

        pos: Tensor = (  # B * 1
            self.pos_head(x).flatten(start_dim=2).argmax(dim=2).flatten(start_dim=1)
        )

        direction: Tensor = (  # B * 2
            self.direction_head(x)
            .flatten(start_dim=2)
            .gather(2, pos.unsqueeze(1).expand(-1, -1, 2))
            .squeeze(1)
        )
        direction[:, 0] %= 2 * torch.pi
        direction[:, 0] -= torch.pi
        direction[:, 1] %= torch.pi
        direction[:, 1] -= torch.pi / 2

        cam_type: Tensor = (  # B * 1
            self.cam_type_head(x)
            .flatten(start_dim=2)
            .gather(2, pos.unsqueeze(1).expand(-1, -1, self.cam_types))
            .squeeze(1)
            .argmax(dim=1, keepdim=True)
        )

        action: Tensor = (  # B * 1
            self.action_head(x)
            .flatten(start_dim=2)
            .gather(2, pos.unsqueeze(1).expand(-1, -1, 4))
            .squeeze(1)
            .argmax(dim=1, keepdim=True)
        )

        return _pos_index2coord(  # B * 7
            observation[0, 0],
            torch.cat((action, pos, direction, cam_type), dim=1),
        )


def _pos_index2coord(occupancy: Tensor, action_with_params: Tensor):
    """
    ## Description:

        Change pos_index in params to pos_coord, making 5-elem params to 7-elem.
    """
    assert action_with_params.ndim == 2 and action_with_params.shape[1] == 5, (
        "`action` should be shaped [B, 5], " f"but got {action_with_params.shape}."
    )
    pos = (occupancy + 1).nonzero()[action_with_params[:, 1].type(torch.int64)]
    action_with_params = torch.cat(
        (
            action_with_params[:, [0]],
            pos,
            action_with_params[:, [2, 3, 4]],
        ),
        dim=1,
    )

    return action_with_params  # B * 7
