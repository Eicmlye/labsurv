from typing import List, OrderedDict

import torch
from labsurv.builders import STRATEGIES
from torch import Tensor
from torch.nn import BatchNorm3d, Conv3d, Module, ReLU, Sequential, Sigmoid


@STRATEGIES.register_module()
class OCPDDPGAddOnlyCleanPolicyNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: str,
        hidden_dim: int,
        action_dim: int,
        cam_types: int,
        neck_layers: int,
        pan_section_num: int = 360,
        tilt_section_num: int = 180,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        self.action_dim = action_dim
        self.pan_section_num = pan_section_num
        self.tilt_section_num = tilt_section_num

        self.neck = Sequential()
        for layer in range(neck_layers):
            cur_layer = Sequential(
                OrderedDict(
                    [
                        (  # conv
                            "conv",
                            Conv3d(
                                in_channels=(
                                    1 if layer == 0 else hidden_dim // 2 ** (layer - 1)
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
        _init_weights(self.pos_head)

        self.direction_head = Sequential(  # B * PAN * W * D * H
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
                            out_channels=(
                                self.pan_section_num * (self.tilt_section_num - 1) + 1
                            ),
                            kernel_size=3,
                            padding=1,
                            dtype=self.FLOAT,
                            device=self.device,
                        ),
                    ),
                ]
            )
        )
        _init_weights(self.direction_head)

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
        _init_weights(self.cam_type_head)

    def forward(self, x: Tensor, pos_mask: Tensor) -> Tensor:
        """
        ## Arguments:

            x (Tensor): [B, 1, W, D, H], 1 for blocked, 2 for visible, 0 for invisible.

            pos_mask (Tensor): [B, 1, W, D, H], pos that allows installation and yet
                haven't been installed at.

        ## Returns:

            action_with_params (Tensor): B * 7.
        """

        assert x.ndim == 5, "Input should be shaped [B, 1, W, D, H]."
        assert (
            str(x.device).startswith("cuda") and str(self.device).startswith("cuda")
        ) or (
            str(x.device).startswith("cpu") and str(self.device).startswith("cpu")
        ), "Different devices found."
        assert x.dtype == self.FLOAT

        occupancy = x.clone().detach()[0, 0] == 1

        x = self.neck(x)

        pos: Tensor = (  # B * 1
            (self.pos_head(x) * pos_mask)
            .flatten(start_dim=2)
            .argmax(dim=2)
            .flatten(start_dim=1)
        )

        direction_index: Tensor = (  # B * 1
            self.direction_head(x)
            .flatten(start_dim=2)
            .gather(2, pos.unsqueeze(1).expand(-1, -1, 1))
            .squeeze(1)
            .argmax(dim=1, keepdim=True)
        )

        # NOTE(eric): When `tilt_index` == 0, `direction` will always
        # pointing to the inversed direction of z axis (the polar point).
        # The polar point will correspond to many pan angles, which results
        # in ambiguous labeling for direction network. So we set `pan_index`
        # to 0 when `tilt_index` is 0.
        pan_index = (direction_index != 0) * (
            (direction_index - 1) % self.pan_section_num
        )
        tilt_index = (direction_index != 0) * (
            (direction_index - 1) // self.tilt_section_num + 1
        )

        pan: Tensor = (
            (pan_index - self.pan_section_num / 2) * 2 * torch.pi / self.pan_section_num
        )
        tilt: Tensor = (
            (tilt_index - self.tilt_section_num / 2) * torch.pi / self.tilt_section_num
        )
        direction: Tensor = torch.cat((pan, tilt), dim=1).type(torch.float32)

        cam_type: Tensor = (  # B * 1
            self.cam_type_head(x)
            .flatten(start_dim=2)
            .gather(2, pos.unsqueeze(1).expand(-1, -1, self.cam_types))
            .squeeze(1)
            .argmax(dim=1, keepdim=True)
        )

        action: Tensor = torch.zeros(
            [pos.shape[0], 1], dtype=torch.float32, device=self.device
        )

        return _pos_index2coord(  # B * 7
            occupancy,
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


def _init_weights(layer: Module, sublayers: List[str] = ["conv", "conv_out"]):
    for sublayer in sublayers:
        torch.nn.init.kaiming_normal_(
            getattr(layer, sublayer).weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(getattr(layer, sublayer).bias, 0)
