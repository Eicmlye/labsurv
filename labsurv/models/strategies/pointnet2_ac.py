"""
PointNet2 mplementation and checkpoint
from https://github.com/yanx27/Pointnet_Pointnet2_pytorch

Network structure ref: https://github.com/charlesq34/pointnet2
"""

from typing import List

import torch
from labsurv.builders import STRATEGIES
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Conv3d,
    Linear,
    Module,
    ModuleList,
    MultiheadAttention,
    ReLU,
    Sequential,
)
from torch.nn.functional import sigmoid

from .pointnet2 import PointNetSetAbstraction


class PointNetBackbone(Module):
    def __init__(self, min_radius: float = 0.1):
        super().__init__()

        self.set_abstraction = ModuleList()

        self.set_abstraction.append(  # [B, 1024, 3], [B, 1024, 64]
            PointNetSetAbstraction(1024, min_radius, 32, 9 + 3, [32, 32, 64], False)
        )
        self.set_abstraction.append(  # [B, 256, 3], [B, 256, 128]
            PointNetSetAbstraction(
                256, 2 * min_radius, 32, 64 + 3, [64, 64, 128], False
            )
        )
        self.set_abstraction.append(  # [B, 64, 3], [B, 64, 256]
            PointNetSetAbstraction(
                64, 4 * min_radius, 32, 128 + 3, [128, 128, 256], False
            )
        )
        self.set_abstraction.append(  # [B, 16, 3], [B, 16, 512]
            PointNetSetAbstraction(
                16, 8 * min_radius, 32, 256 + 3, [256, 256, 512], False
            )
        )

    def forward(self, input_data: Tensor):
        """
        ## Arguments:

            input_data (Tensor): [B, N, DATA_DIM]

        ## Returns:

            coords (List[Tensor])

            data (List[Tensor])
        """
        coords: List[Tensor] = [input_data[:, :, :3]]  # [B, N, 3]
        data: List[Tensor] = [input_data]  # [B, N, DATA_DIM]

        for index in range(len(self.set_abstraction)):
            dsampled_coords, dsampled_data = self.set_abstraction[index](
                coords[-1], data[-1]
            )
            coords.append(dsampled_coords)
            data.append(dsampled_data)

        return coords, data


@STRATEGIES.register_module()
class PointNet2Actor(Module):
    def __init__(
        self,
        device: str,
        hidden_dim: int,
        cam_types: int,
        attn_head_num: int = 4,
        min_radius: float = 0.1,
        normalize_input: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        self.normalize_input = normalize_input

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]
        # [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        action_dim = 10 + self.cam_types

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim + 1, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, 512, device=self.device),
            ReLU(),
        )

        # neighbourhood
        self.backbone = PointNetBackbone(min_radius)
        self.neigh_attn = [
            MultiheadAttention(
                embed_dim=64, num_heads=attn_head_num, device=self.device
            ),
            MultiheadAttention(
                embed_dim=128, num_heads=attn_head_num, device=self.device
            ),
            MultiheadAttention(
                embed_dim=256, num_heads=attn_head_num, device=self.device
            ),
            MultiheadAttention(
                embed_dim=512, num_heads=attn_head_num, device=self.device
            ),
        ]
        self.neigh_ff = [
            Sequential(
                Linear(64, 64, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(128, 128, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(256, 256, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(512, 512, device=self.device),
                ReLU(),
            ),
        ]
        self.neigh_upsample = [
            Sequential(
                Linear(64, 32, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(128, 64, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(256, 128, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(512, 256, device=self.device),
                ReLU(),
            ),
        ]

        # out
        self.out = Sequential(
            Linear(32, 32, device=self.device),
            ReLU(),
            Linear(32, action_dim, device=self.device),
        )

    def forward(
        self,
        self_and_neigh_params: Tensor,
        self_mask: Tensor,
        neigh: Tensor,
        voxel_length: float,
        room_shape: List[int],
    ) -> Tensor:
        """
        ## Arguments:

            self_and_neigh_params (Tensor): [B, AGENT_NUM(NEIGH), PARAM_DIM], torch.float.
            Params of agents in `neigh` including current agent itself.

            self_mask (Tensor): [B, AGENT_NUM(NEIGH)], torch.bool. 1 for current agent, 0
            for neighbours.

            neigh (Tensor): [B, 3, 2L+1, 2L+1, 2L+1], torch.float, where `L` is the
            farther dof of the camera. `occupancy`, `install_permitted`,
            `vis_redundancy`. `vis_redundancy` = vis_count / agent_num, -1 if not
            in `must_monitor`.

        ## Returns:

            dist (Tensor): [B, ACTION_DIM], torch.float, action distribution.
            [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        """

        ## params processing
        batch_size = self_mask.shape[0]
        neigh_agent_num = self_mask.shape[1]
        agent_params = (
            _normalize_param(self_and_neigh_params, room_shape)
            if self.normalize_input
            else self_and_neigh_params
        )

        agent_params_with_pos = torch.concat(
            (agent_params, self_mask.unsqueeze(-1)), dim=-1
        )
        # [B, AGENT_NUM(NEIGH), 512]
        agent_embedding: Tensor = self.param_encoder(agent_params_with_pos)

        ## neighbourhood pointcloud processing
        input_data = _ply2data(neigh, voxel_length)  # [B, N, DATA_DIM]
        # [B, SA_SAMPLE_NUM, 64/128/256/512]
        data = self.backbone(input_data)[1][1:]
        # [SA_SAMPLE_NUM, B, 64/128/256/512]
        data = [item.permute(1, 0, 2) for item in data[1:]]

        # [AGENT_NUM(NEIGH), B, 512/256/128/64/32]
        attn_outputs = [agent_embedding.permute(1, 0, 2)]
        for i in reversed(range(len(self.neigh_attn))):
            # [AGENT_NUM(NEIGH), B, 512/256/128/64]
            attn_feats: Tensor = self.neigh_attn[i](
                query=attn_outputs[-1],
                key=data[i],
                value=data[i],
            )[0].squeeze(dim=0)
            attn_feats += attn_outputs[-1]  # skip connection
            # [AGENT_NUM(NEIGH), B, 512/256/128/64]
            attn_feats: Tensor = (
                self.neigh_ff[i](
                    attn_feats.view(batch_size * neigh_agent_num, -1)
                ).view(neigh_agent_num, batch_size, -1)
                + attn_feats
            )
            attn_feats: Tensor = self.neigh_upsample[i](
                attn_feats.view(batch_size * neigh_agent_num, -1)
            ).view(neigh_agent_num, batch_size, -1)
            attn_outputs.append(attn_feats)

        # out
        out: Tensor = self.out(
            attn_outputs[-1]
            .permute(1, 0, 2)[self_mask.unsqueeze(-1).repeat(1, 1, 32)]
            .view(-1, 32)
        )
        return out


@STRATEGIES.register_module()
class PointNet2Critic(Module):
    def __init__(
        self,
        device: str,
        hidden_dim: int,
        cam_types: int,
        attn_head_num: int = 4,
        min_radius: float = 0.1,
        normalize_input: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        self.normalize_input = normalize_input

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim + 1, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, 512, device=self.device),
            ReLU(),
        )

        # neighbourhood
        self.backbone = PointNetBackbone(min_radius)
        self.neigh_attn = [
            MultiheadAttention(
                embed_dim=64, num_heads=attn_head_num, device=self.device
            ),
            MultiheadAttention(
                embed_dim=128, num_heads=attn_head_num, device=self.device
            ),
            MultiheadAttention(
                embed_dim=256, num_heads=attn_head_num, device=self.device
            ),
            MultiheadAttention(
                embed_dim=512, num_heads=attn_head_num, device=self.device
            ),
        ]
        self.neigh_ff = [
            Sequential(
                Linear(64, 64, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(128, 128, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(256, 256, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(512, 512, device=self.device),
                ReLU(),
            ),
        ]
        self.neigh_upsample = [
            Sequential(
                Linear(64, 32, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(128, 64, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(256, 128, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(512, 256, device=self.device),
                ReLU(),
            ),
        ]

        # out
        self.out = Sequential(
            Linear(32, 32, device=self.device),
            ReLU(),
            Linear(32, 1, device=self.device),
        )

    def forward(
        self,
        cam_params: Tensor,
        env: Tensor,
        voxel_length: float,
        room_shape: List[int],
    ) -> Tensor:
        """
        ## Arguments:

            cam_params (Tensor): [B, AGENT_NUM, PARAM_DIM], torch.float. Absolute
            coords and absolute angles, one-hot cam_type vecs.

            env (Tensor): [B, 3, W, D, H], torch.float, `occupancy`,
            `install_permitted`, `vis_redundancy`.
            `vis_redundancy` = vis_count / agent_num, -1 if not in `must_monitor`.

        ## Returns:

            value_predicted (Tensor): [B, 1], torch.float.
        """

        ## params processing
        batch_size = cam_params.shape[0]
        neigh_agent_num = cam_params.shape[1]
        agent_params = (
            _normalize_param(cam_params, room_shape)
            if self.normalize_input
            else cam_params
        )

        # [B, AGENT_NUM(NEIGH), 512]
        agent_embedding: Tensor = self.param_encoder(agent_params)
        ## neighbourhood pointcloud processing
        input_data = _ply2data(env, voxel_length)  # [B, N, DATA_DIM]
        # [B, SA_SAMPLE_NUM, 64/128/256/512]
        data = self.backbone(input_data)[1][1:]
        # [SA_SAMPLE_NUM, B, 64/128/256/512]
        data = [item.permute(1, 0, 2) for item in data[1:]]

        # [AGENT_NUM(NEIGH), B, 512/256/128/64/32]
        attn_outputs = [agent_embedding.permute(1, 0, 2)]
        for i in reversed(range(len(self.neigh_attn))):
            # [AGENT_NUM(NEIGH), B, 512/256/128/64]
            attn_feats: Tensor = self.neigh_attn[i](
                query=attn_outputs[-1],
                key=data[i],
                value=data[i],
            )[0].squeeze(dim=0)
            attn_feats += attn_outputs[-1]  # skip connection
            # [AGENT_NUM(NEIGH), B, 512/256/128/64]
            attn_feats: Tensor = (
                self.neigh_ff[i](
                    attn_feats.view(batch_size * neigh_agent_num, -1)
                ).view(neigh_agent_num, batch_size, -1)
                + attn_feats
            )
            attn_feats: Tensor = self.neigh_upsample[i](
                attn_feats.view(batch_size * neigh_agent_num, -1)
            ).view(neigh_agent_num, batch_size, -1)
            attn_outputs.append(attn_feats)

        # out
        value_predicted = self.out(attn_outputs[-1].permute(1, 0, 2)).mean(dim=1)

        return value_predicted


@STRATEGIES.register_module()
class SimplePointNet2Actor(Module):
    def __init__(
        self,
        device: str,
        hidden_dim: int,
        cam_types: int,
        attn_head_num: int = 4,
        min_radius: float = 0.1,
        normalize_input: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        self.normalize_input = normalize_input

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]
        # [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        action_dim = 10 + self.cam_types

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim + 1, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, 512, device=self.device),
            ReLU(),
        )

        # neighbourhood
        self.backbone = PointNetBackbone(min_radius)
        self.neigh_attn = MultiheadAttention(
            embed_dim=512, num_heads=attn_head_num, device=self.device
        )
        self.neigh_ff = Sequential(
            Linear(512, 512, device=self.device),
            ReLU(),
        )
        self.neigh_mlp = Sequential(
            Linear(512, 256, device=self.device),
            ReLU(),
            Linear(256, 64, device=self.device),
            ReLU(),
        )

        # out
        self.out = Sequential(
            Linear(64, 32, device=self.device),
            ReLU(),
            Linear(32, action_dim, device=self.device),
        )

    def forward(
        self,
        self_and_neigh_params: Tensor,
        self_mask: Tensor,
        neigh: Tensor,
        voxel_length: float,
        room_shape: List[int],
    ) -> Tensor:
        """
        ## Arguments:

            self_and_neigh_params (Tensor): [B, AGENT_NUM(NEIGH), PARAM_DIM], torch.float.
            Params of agents in `neigh` including current agent itself.

            self_mask (Tensor): [B, AGENT_NUM(NEIGH)], torch.bool. 1 for current agent, 0
            for neighbours.

            neigh (Tensor): [B, 3, 2L+1, 2L+1, 2L+1], torch.float, where `L` is the
            farther dof of the camera. `occupancy`, `install_permitted`,
            `vis_redundancy`. `vis_redundancy` = vis_count / agent_num, -1 if not
            in `must_monitor`.

        ## Returns:

            dist (Tensor): [B, ACTION_DIM], torch.float, action distribution.
            [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        """

        ## params processing
        batch_size = self_mask.shape[0]
        neigh_agent_num = self_mask.shape[1]
        agent_params = (
            _normalize_param(self_and_neigh_params, room_shape)
            if self.normalize_input
            else self_and_neigh_params
        )

        agent_params_with_pos = torch.concat(
            (agent_params, self_mask.unsqueeze(-1)), dim=-1
        )
        # [B, AGENT_NUM(NEIGH), 512]
        agent_embedding: Tensor = self.param_encoder(agent_params_with_pos)

        ## neighbourhood pointcloud processing
        input_data = _ply2data(neigh, voxel_length)  # [B, N, DATA_DIM]
        # [B, SA_SAMPLE_NUM, 64/128/256/512]
        data = self.backbone(input_data)[1][1:]
        # [SA_SAMPLE_NUM, B, 512]
        data = data[-1].permute(1, 0, 2)

        # [AGENT_NUM(NEIGH), B, 512]
        attn_feats: Tensor = self.neigh_attn(
            query=agent_embedding.permute(1, 0, 2),
            key=data,
            value=data,
        )[0].squeeze(dim=0)
        attn_feats += agent_embedding.permute(1, 0, 2)
        # [AGENT_NUM(NEIGH), B, 512]
        attn_feats: Tensor = (
            self.neigh_ff(attn_feats.view(batch_size * neigh_agent_num, -1)).view(
                neigh_agent_num, batch_size, -1
            )
            + attn_feats
        )

        # [AGENT_NUM(NEIGH), B, 64]
        mlp_feats: Tensor = self.neigh_mlp(
            attn_feats.view(batch_size * neigh_agent_num, -1)
        ).view(neigh_agent_num, batch_size, -1)

        # out
        out: Tensor = self.out(
            mlp_feats[-1]
            .permute(1, 0, 2)[self_mask.unsqueeze(-1).repeat(1, 1, 64)]
            .view(-1, 64)
        )

        return out


@STRATEGIES.register_module()
class SimplePointNet2Critic(Module):
    def __init__(
        self,
        device: str,
        hidden_dim: int,
        cam_types: int,
        attn_head_num: int = 4,
        min_radius: float = 0.1,
        normalize_input: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        self.normalize_input = normalize_input

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim + 1, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, 512, device=self.device),
            ReLU(),
        )

        # neighbourhood
        self.backbone = PointNetBackbone(min_radius)
        self.neigh_attn = MultiheadAttention(
            embed_dim=512, num_heads=attn_head_num, device=self.device
        )
        self.neigh_ff = Sequential(
            Linear(512, 512, device=self.device),
            ReLU(),
        )
        self.neigh_mlp = Sequential(
            Linear(512, 256, device=self.device),
            ReLU(),
            Linear(256, 64, device=self.device),
            ReLU(),
        )

        # out
        self.out = Sequential(
            Linear(64, 32, device=self.device),
            ReLU(),
            Linear(32, 1, device=self.device),
        )

    def forward(
        self,
        cam_params: Tensor,
        env: Tensor,
        voxel_length: float,
        room_shape: List[int],
    ) -> Tensor:
        """
        ## Arguments:

            cam_params (Tensor): [B, AGENT_NUM, PARAM_DIM], torch.float. Absolute
            coords and absolute angles, one-hot cam_type vecs.

            env (Tensor): [B, 3, W, D, H], torch.float, `occupancy`,
            `install_permitted`, `vis_redundancy`.
            `vis_redundancy` = vis_count / agent_num, -1 if not in `must_monitor`.

        ## Returns:

            value_predicted (Tensor): [B, 1], torch.float.
        """

        ## params processing
        batch_size = cam_params.shape[0]
        neigh_agent_num = cam_params.shape[1]
        agent_params = (
            _normalize_param(cam_params, room_shape)
            if self.normalize_input
            else cam_params
        )

        # [B, AGENT_NUM(NEIGH), 512]
        agent_embedding: Tensor = self.param_encoder(agent_params)

        ## neighbourhood pointcloud processing
        input_data = _ply2data(env, voxel_length)  # [B, N, DATA_DIM]
        # [B, SA_SAMPLE_NUM, 64/128/256/512]
        data = self.backbone(input_data)[1][1:]
        # [SA_SAMPLE_NUM, B, 512]
        data = data[-1].permute(1, 0, 2)

        # [AGENT_NUM(NEIGH), B, 512]
        attn_feats: Tensor = self.neigh_attn(
            query=agent_embedding.permute(1, 0, 2),
            key=data,
            value=data,
        )[0].squeeze(dim=0)
        attn_feats += agent_embedding.permute(1, 0, 2)
        # [AGENT_NUM(NEIGH), B, 512]
        attn_feats: Tensor = (
            self.neigh_ff(attn_feats.view(batch_size * neigh_agent_num, -1)).view(
                neigh_agent_num, batch_size, -1
            )
            + attn_feats
        )

        # [AGENT_NUM(NEIGH), B, 64]
        mlp_feats: Tensor = self.neigh_mlp(
            attn_feats.view(batch_size * neigh_agent_num, -1)
        ).view(neigh_agent_num, batch_size, -1)

        # out
        value_predicted = self.out(mlp_feats[-1].permute(1, 0, 2)).mean(dim=1)

        return value_predicted


@STRATEGIES.register_module()
class ConvActor(Module):
    def __init__(
        self,
        device: str,
        hidden_dim: int,
        cam_types: int,
        attn_head_num: int = 4,
        normalize_input: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        self.normalize_input = normalize_input

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]
        # [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        action_dim = 10 + self.cam_types

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim + 1, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, 512, device=self.device),
            ReLU(),
        )

        # neighbourhood
        self.backbone = [
            Sequential(
                Conv3d(3, 64, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(64, 128, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(128, 256, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(256, 512, 3, 1, 1, device=self.device),
                ReLU(),
            ),
        ]
        self.neigh_attn = [
            MultiheadAttention(
                embed_dim=64, num_heads=attn_head_num, device=self.device
            ),
            MultiheadAttention(
                embed_dim=128, num_heads=attn_head_num, device=self.device
            ),
            MultiheadAttention(
                embed_dim=256, num_heads=attn_head_num, device=self.device
            ),
            MultiheadAttention(
                embed_dim=512, num_heads=attn_head_num, device=self.device
            ),
        ]
        self.neigh_ff = [
            Sequential(
                Linear(64, 64, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(128, 128, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(256, 256, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(512, 512, device=self.device),
                ReLU(),
            ),
        ]
        self.neigh_upsample = [
            Sequential(
                Linear(64, 32, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(128, 64, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(256, 128, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(512, 256, device=self.device),
                ReLU(),
            ),
        ]

        # out
        self.out = Sequential(
            Linear(32, 32, device=self.device),
            ReLU(),
            Linear(32, action_dim, device=self.device),
        )

    def forward(
        self,
        self_and_neigh_params: Tensor,
        self_mask: Tensor,
        neigh: Tensor,
        voxel_length: float,
        room_shape: List[int],
    ) -> Tensor:
        """
        ## Arguments:

            self_and_neigh_params (Tensor): [B, AGENT_NUM(NEIGH), PARAM_DIM], torch.float.
            Params of agents in `neigh` including current agent itself.

            self_mask (Tensor): [B, AGENT_NUM(NEIGH)], torch.bool. 1 for current agent, 0
            for neighbours.

            neigh (Tensor): [B, 3, 2L+1, 2L+1, 2L+1], torch.float, where `L` is the
            farther dof of the camera. `occupancy`, `install_permitted`,
            `vis_redundancy`. `vis_redundancy` = vis_count / agent_num, -1 if not
            in `must_monitor`.

        ## Returns:

            dist (Tensor): [B, ACTION_DIM], torch.float, action distribution.
            [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        """

        ## params processing
        batch_size = self_mask.shape[0]
        neigh_agent_num = self_mask.shape[1]
        agent_params = (
            _normalize_param(self_and_neigh_params, room_shape)
            if self.normalize_input
            else self_and_neigh_params
        )

        agent_params_with_pos = torch.concat(
            (agent_params, self_mask.unsqueeze(-1)), dim=-1
        )
        # [B, AGENT_NUM(NEIGH), 512]
        agent_embedding: Tensor = self.param_encoder(agent_params_with_pos)

        ## neighbourhood pointcloud processing
        unet_input = [neigh]  # [B, 3/64/128/256/512, 2L+1, 2L+1, 2L+1]
        for i in range(4):
            conv_feats = self.backbone[i](unet_input[-1])
            unet_input.append(conv_feats)
        unet_input = unet_input[1:]  # [B, 64/128/256/512, 2L+1, 2L+1, 2L+1]

        # [AGENT_NUM(NEIGH), B, 512/256/128/64/32]
        attn_outputs = [agent_embedding.permute(1, 0, 2)]
        for i in reversed(range(len(self.neigh_attn))):
            # [AGENT_NUM(NEIGH), B, 512/256/128/64]
            attn_feats: Tensor = self.neigh_attn[i](
                query=attn_outputs[-1],
                key=unet_input[i].flatten(start_dim=2).permute(2, 0, 1),
                value=unet_input[i].flatten(start_dim=2).permute(2, 0, 1),
            )[0].squeeze(dim=0)
            attn_feats += attn_outputs[-1]  # skip connection
            # [AGENT_NUM(NEIGH), B, 512/256/128/64]
            attn_feats: Tensor = (
                self.neigh_ff[i](
                    attn_feats.view(batch_size * neigh_agent_num, -1)
                ).view(neigh_agent_num, batch_size, -1)
                + attn_feats
            )
            attn_feats: Tensor = self.neigh_upsample[i](
                attn_feats.view(batch_size * neigh_agent_num, -1)
            ).view(neigh_agent_num, batch_size, -1)
            attn_outputs.append(attn_feats)

        # out
        out: Tensor = self.out(
            attn_outputs[-1]
            .permute(1, 0, 2)[self_mask.unsqueeze(-1).repeat(1, 1, 32)]
            .view(-1, 32)
        )

        return out


@STRATEGIES.register_module()
class ConvCritic(Module):
    def __init__(
        self,
        device: str,
        hidden_dim: int,
        cam_types: int,
        attn_head_num: int = 4,
        normalize_input: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        self.normalize_input = normalize_input

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim, hidden_dim // 2, device=self.device),
            ReLU(),
            Linear(hidden_dim // 2, hidden_dim, device=self.device),
            ReLU(),
        )

        # neighbourhood
        self.backbone = [
            Sequential(
                Conv3d(3, 64, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(64, 128, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(128, 256, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(256, 512, 3, 1, 1, device=self.device),
                ReLU(),
            ),
        ]
        self.neigh_attn = [
            MultiheadAttention(
                embed_dim=64, num_heads=attn_head_num, device=self.device
            ),
            MultiheadAttention(
                embed_dim=128, num_heads=attn_head_num, device=self.device
            ),
            MultiheadAttention(
                embed_dim=256, num_heads=attn_head_num, device=self.device
            ),
            MultiheadAttention(
                embed_dim=512, num_heads=attn_head_num, device=self.device
            ),
        ]
        self.neigh_ff = [
            Sequential(
                Linear(64, 64, device=self.device),
                BatchNorm1d(64, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(128, 128, device=self.device),
                BatchNorm1d(128, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(256, 256, device=self.device),
                BatchNorm1d(256, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(512, 512, device=self.device),
                BatchNorm1d(512, device=self.device),
                ReLU(),
            ),
        ]
        self.neigh_upsample = [
            Sequential(
                Linear(64, 32, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(128, 64, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(256, 128, device=self.device),
                ReLU(),
            ),
            Sequential(
                Linear(512, 256, device=self.device),
                ReLU(),
            ),
        ]

        # out
        self.out = Sequential(
            Linear(32, 32, device=self.device),
            ReLU(),
            Linear(32, 1, device=self.device),
        )

    def forward(
        self,
        cam_params: Tensor,
        env: Tensor,
        voxel_length: float,
        room_shape: List[int],
    ) -> Tensor:
        """
        ## Arguments:

            cam_params (Tensor): [B, AGENT_NUM, PARAM_DIM], torch.float. Absolute
            coords and absolute angles, one-hot cam_type vecs.

            env (Tensor): [B, 3, W, D, H], torch.float, `occupancy`,
            `install_permitted`, `vis_redundancy`.
            `vis_redundancy` = vis_count / agent_num, -1 if not in `must_monitor`.

        ## Returns:

            value_predicted (Tensor): [B, 1], torch.float.
        """

        ## params processing
        batch_size = cam_params.shape[0]
        neigh_agent_num = cam_params.shape[1]
        agent_params = (
            _normalize_param(cam_params, room_shape)
            if self.normalize_input
            else cam_params
        )

        # [B, AGENT_NUM(NEIGH), 512]
        agent_embedding: Tensor = self.param_encoder(agent_params)
        ## neighbourhood pointcloud processing
        unet_input = [env]  # [B, 3/64/128/256/512, 2L+1, 2L+1, 2L+1]
        for i in range(4):
            conv_feats = self.backbone[i](unet_input[-1])
            unet_input.append(conv_feats)
        unet_input = unet_input[1:]  # [B, 64/128/256/512, 2L+1, 2L+1, 2L+1]

        # [AGENT_NUM(NEIGH), B, 512/256/128/64/32]
        attn_outputs = [agent_embedding.permute(1, 0, 2)]
        for i in reversed(range(4)):
            # [AGENT_NUM(NEIGH), B, 512/256/128/64]
            attn_feats: Tensor = self.neigh_attn[i](
                query=attn_outputs[-1],
                key=unet_input[i].flatten(start_dim=2).permute(2, 0, 1),
                value=unet_input[i].flatten(start_dim=2).permute(2, 0, 1),
            )[0].squeeze(dim=0)
            attn_feats += attn_outputs[-1]  # skip connection
            # [AGENT_NUM(NEIGH), B, 512/256/128/64]
            attn_feats: Tensor = (
                self.neigh_ff[i](
                    attn_feats.view(batch_size * neigh_agent_num, -1)
                ).view(neigh_agent_num, batch_size, -1)
                + attn_feats
            )
            attn_feats: Tensor = self.neigh_upsample[i](
                attn_feats.view(batch_size * neigh_agent_num, -1)
            ).view(neigh_agent_num, batch_size, -1)
            attn_outputs.append(attn_feats)

        # out
        value_predicted = self.out(attn_outputs[-1].permute(1, 0, 2)).mean(dim=1)

        return value_predicted


@STRATEGIES.register_module()
class SimpleConvActor(Module):
    def __init__(
        self,
        device: str,
        hidden_dim: int,
        cam_types: int,
        attn_head_num: int = 4,
        normalize_input: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        self.normalize_input = normalize_input

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]
        # [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        action_dim = 10 + self.cam_types

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim + 1, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, 64, device=self.device),
            ReLU(),
        )
        self.param_mlp = Sequential(
            Linear(64, 512, device=self.device),
            ReLU(),
            Linear(512, 512, device=self.device),
            ReLU(),
        )

        # neighbourhood
        self.backbone = [
            Sequential(
                Conv3d(3, 64, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(64, 128, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(128, 256, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(256, 512, 3, 1, 1, device=self.device),
                ReLU(),
            ),
        ]
        self.neigh_attn = MultiheadAttention(
            embed_dim=512, num_heads=attn_head_num, device=self.device
        )
        self.neigh_ff = Sequential(
            Linear(512, 512, device=self.device),
            ReLU(),
        )
        self.neigh_mlp = Sequential(
            Linear(512, 512, device=self.device),
            ReLU(),
            Linear(512, 64, device=self.device),
            ReLU(),
        )

        # out
        self.comm = Sequential(
            Linear(64, 64, device=self.device),
            ReLU(),
            Linear(64, action_dim, device=self.device),
        )
        self.out = Sequential(
            Linear(64, 64, device=self.device),
            ReLU(),
            Linear(64, action_dim, device=self.device),
        )

    def forward(
        self,
        self_and_neigh_params: Tensor,
        self_mask: Tensor,
        neigh: Tensor,
        voxel_length: float,
        room_shape: List[int],
    ) -> Tensor:
        """
        ## Arguments:

            self_and_neigh_params (Tensor): [B, AGENT_NUM(NEIGH), PARAM_DIM], torch.float.
            Params of agents in `neigh` including current agent itself.

            self_mask (Tensor): [B, AGENT_NUM(NEIGH)], torch.bool. 1 for current agent, 0
            for neighbours.

            neigh (Tensor): [B, 3, 2L+1, 2L+1, 2L+1], torch.float, where `L` is the
            farther dof of the camera. `occupancy`, `install_permitted`,
            `vis_redundancy`. `vis_redundancy` = vis_count / agent_num, -1 if not
            in `must_monitor`.

        ## Returns:

            dist (Tensor): [B, ACTION_DIM], torch.float, action distribution.
            [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        """

        ## params processing
        batch_size = self_mask.shape[0]
        neigh_agent_num = self_mask.shape[1]
        agent_params = (
            _normalize_param(self_and_neigh_params, room_shape)
            if self.normalize_input
            else self_and_neigh_params
        )

        agent_params_with_pos = torch.concat(
            (agent_params, self_mask.unsqueeze(-1)), dim=-1
        )
        # [B, AGENT_NUM(NEIGH), 64]
        agent_embedding: Tensor = self.param_encoder(agent_params_with_pos)
        # [B, AGENT_NUM(NEIGH), 512]
        agent_output: Tensor = self.param_mlp(agent_embedding)

        ## neighbourhood pointcloud processing
        unet_input = [neigh]  # [B, 3/64/128/256/512, 2L+1, 2L+1, 2L+1]
        for i in range(4):
            conv_feats = self.backbone[i](unet_input[-1])
            unet_input.append(conv_feats)
        unet_output = unet_input[-1]  # [B, 512, 2L+1, 2L+1, 2L+1]

        # [AGENT_NUM(NEIGH), B, 512]
        attn_feats: Tensor = self.neigh_attn(
            query=agent_output.permute(1, 0, 2),
            key=unet_output.flatten(start_dim=2).permute(2, 0, 1),
            value=unet_output.flatten(start_dim=2).permute(2, 0, 1),
        )[0]
        attn_feats += agent_output.permute(1, 0, 2)
        # [AGENT_NUM(NEIGH), B, 512]
        attn_feats: Tensor = (
            self.neigh_ff(attn_feats.view(batch_size * neigh_agent_num, -1)).view(
                neigh_agent_num, batch_size, -1
            )
            + attn_feats
        )

        # [AGENT_NUM(NEIGH), B, 64]
        mlp_feats: Tensor = self.neigh_mlp(
            attn_feats.view(batch_size * neigh_agent_num, -1)
        ).view(neigh_agent_num, batch_size, -1)
        # NOTE(eric): don't know why but should not use += here
        # may raise inplace operation error
        mlp_feats = mlp_feats + agent_embedding.permute(1, 0, 2)

        # out
        comm: Tensor = self.comm(
            mlp_feats.permute(1, 0, 2)[
                torch.logical_not(self_mask).unsqueeze(-1).repeat(1, 1, 64)
            ].view(-1, neigh_agent_num - 1, 64)
        ).mean(dim=1)
        out: Tensor = self.out(
            mlp_feats.permute(1, 0, 2)[self_mask.unsqueeze(-1).repeat(1, 1, 64)].view(
                -1, 64
            )
        )

        return out + comm


@STRATEGIES.register_module()
class SimpleConvCritic(Module):
    def __init__(
        self,
        device: str,
        hidden_dim: int,
        cam_types: int,
        attn_head_num: int = 4,
        normalize_input: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        self.normalize_input = normalize_input

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, 64, device=self.device),
            ReLU(),
        )
        self.param_mlp = Sequential(
            Linear(64, 512, device=self.device),
            ReLU(),
            Linear(512, 512, device=self.device),
            ReLU(),
        )

        # neighbourhood
        self.backbone = [
            Sequential(
                Conv3d(3, 64, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(64, 128, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(128, 256, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(256, 512, 3, 1, 1, device=self.device),
                ReLU(),
            ),
        ]
        self.neigh_attn = MultiheadAttention(
            embed_dim=512, num_heads=attn_head_num, device=self.device
        )
        self.neigh_ff = Sequential(
            Linear(512, 512, device=self.device),
            ReLU(),
        )
        self.neigh_mlp = Sequential(
            Linear(512, 512, device=self.device),
            ReLU(),
            Linear(512, 64, device=self.device),
            ReLU(),
        )

        # out
        self.out = Sequential(
            Linear(64, 64, device=self.device),
            ReLU(),
            Linear(64, 1, device=self.device),
        )

    def forward(
        self,
        cam_params: Tensor,
        env: Tensor,
        voxel_length: float,
        room_shape: List[int],
    ) -> Tensor:
        """
        ## Arguments:

            cam_params (Tensor): [B, AGENT_NUM, PARAM_DIM], torch.float. Absolute
            coords and absolute angles, one-hot cam_type vecs.

            env (Tensor): [B, 3, W, D, H], torch.float, `occupancy`,
            `install_permitted`, `vis_redundancy`.
            `vis_redundancy` = vis_count / agent_num, -1 if not in `must_monitor`.

        ## Returns:

            value_predicted (Tensor): [B, 1], torch.float.
        """

        ## params processing
        batch_size = cam_params.shape[0]
        neigh_agent_num = cam_params.shape[1]
        agent_params = (
            _normalize_param(cam_params, room_shape)
            if self.normalize_input
            else cam_params
        )

        # [B, AGENT_NUM(NEIGH), 64]
        agent_embedding: Tensor = self.param_encoder(agent_params)
        # [B, AGENT_NUM(NEIGH), 512]
        agent_input: Tensor = self.param_mlp(agent_embedding)

        ## neighbourhood pointcloud processing
        unet_input = [env]  # [B, 3/64/128/256/512, 2L+1, 2L+1, 2L+1]
        for i in range(4):
            conv_feats = self.backbone[i](unet_input[-1])
            unet_input.append(conv_feats)
        unet_output = unet_input[-1]  # [B, 512, 2L+1, 2L+1, 2L+1]

        # [AGENT_NUM(NEIGH), B, 512]
        attn_feats: Tensor = self.neigh_attn(
            query=agent_input.permute(1, 0, 2),
            key=unet_output.flatten(start_dim=2).permute(2, 0, 1),
            value=unet_output.flatten(start_dim=2).permute(2, 0, 1),
        )[0]
        attn_feats += agent_input.permute(1, 0, 2)
        # [AGENT_NUM(NEIGH), B, 512]
        attn_feats: Tensor = (
            self.neigh_ff(attn_feats.view(batch_size * neigh_agent_num, -1)).view(
                neigh_agent_num, batch_size, -1
            )
            + attn_feats
        )

        # [AGENT_NUM(NEIGH), B, 64]
        mlp_feats: Tensor = self.neigh_mlp(
            attn_feats.view(batch_size * neigh_agent_num, -1)
        ).view(neigh_agent_num, batch_size, -1)
        # NOTE(eric): don't know why but should not use += here
        # may raise inplace operation error
        mlp_feats = mlp_feats + agent_embedding.permute(1, 0, 2)

        # out
        value_predicted = self.out(mlp_feats.permute(1, 0, 2)).mean(dim=1)

        return value_predicted


@STRATEGIES.register_module()
class SimpleConvCommActor(Module):
    def __init__(
        self,
        device: str,
        hidden_dim: int,
        cam_types: int,
        attn_head_num: int = 4,
        normalize_input: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        self.normalize_input = normalize_input

        self.param_dim = (
            5 + self.cam_types
        )  # [x, y, z, pan, tilt, (one-hot cam_type vec)]
        # [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        action_dim = 10 + self.cam_types

        # self param
        self.param_encoder = Sequential(
            Linear(self.param_dim + 1, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, 64, device=self.device),
            ReLU(),
        )
        self.mes_attn = MultiheadAttention(
            embed_dim=64, num_heads=1, device=self.device
        )
        self.mes_mlp = Sequential(
            Linear(64, 16, device=self.device),
            ReLU(),
            Linear(16, 64, device=self.device),
            ReLU(),
        )
        self.param_mlp = Sequential(
            Linear(64, 512, device=self.device),
            ReLU(),
            Linear(512, 512, device=self.device),
            ReLU(),
        )

        # neighbourhood
        self.backbone = [
            Sequential(
                Conv3d(3, 64, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(64, 128, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(128, 256, 3, 1, 1, device=self.device),
                ReLU(),
            ),
            Sequential(
                Conv3d(256, 512, 3, 1, 1, device=self.device),
                ReLU(),
            ),
        ]
        self.neigh_attn = MultiheadAttention(
            embed_dim=512, num_heads=attn_head_num, device=self.device
        )
        self.neigh_ff = Sequential(
            Linear(512, 512, device=self.device),
            ReLU(),
        )
        self.neigh_mlp = Sequential(
            Linear(512, 256, device=self.device),
            ReLU(),
            Linear(256, 64, device=self.device),
            ReLU(),
        )

        # out
        self.implicit_comm = Sequential(
            Linear(64, 32, device=self.device),
            ReLU(),
            Linear(32, action_dim, device=self.device),
        )
        self.out = Sequential(
            Linear(64, 32, device=self.device),
            ReLU(),
            Linear(32, action_dim, device=self.device),
        )
        self.explicit_comm = Sequential(
            Linear(64, 16, device=self.device),
            ReLU(),
            Linear(16, 64, device=self.device),
        )

    def forward(
        self,
        self_and_neigh_params_with_mes: Tensor,
        self_mask: Tensor,
        neigh: Tensor,
        voxel_length: float,
        room_shape: List[int],
    ) -> Tensor:
        """
        ## Arguments:

            self_and_neigh_params_with_mes (Tensor):
            [B, AGENT_NUM(NEIGH), PARAM_DIM+64], torch.float. Params of agents
            in `neigh` including current agent itself.

            self_mask (Tensor): [B, AGENT_NUM(NEIGH)], torch.bool. 1 for current agent, 0
            for neighbours.

            neigh (Tensor): [B, 3, 2L+1, 2L+1, 2L+1], torch.float, where `L` is the
            farther dof of the camera. `occupancy`, `install_permitted`,
            `vis_redundancy`. `vis_redundancy` = vis_count / agent_num, -1 if not
            in `must_monitor`.

        ## Returns:

            dist (Tensor): [B, ACTION_DIM], torch.float, action distribution.
            [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        """

        ## params processing
        batch_size = self_mask.shape[0]
        neigh_agent_num = self_mask.shape[1]

        self_and_neigh_params = self_and_neigh_params_with_mes[:, :, : self.param_dim]
        if self_and_neigh_params_with_mes.shape[-1] == self.param_dim + 64:
            messages = self_and_neigh_params_with_mes[:, :, self.param_dim :]
        elif self_and_neigh_params_with_mes.shape[-1] == self.param_dim:
            messages = torch.zeros(
                self_and_neigh_params_with_mes.shape[0],
                self_and_neigh_params_with_mes.shape[1],
                64,
            )
        else:
            raise ValueError(
                "Illegal message length found. Expected 64, got "
                f"{self_and_neigh_params_with_mes.shape[-1] - self.param_dim}."
            )
        agent_params = (
            _normalize_param(self_and_neigh_params, room_shape)
            if self.normalize_input
            else self_and_neigh_params
        )

        agent_params_with_pos = torch.concat(
            (agent_params, self_mask.unsqueeze(-1)), dim=-1
        )
        # [B, AGENT_NUM(NEIGH), 64]
        agent_embedding: Tensor = self.param_encoder(agent_params_with_pos)
        mes_embedding: Tensor = self.mes_attn(
            query=agent_embedding.permute(1, 0, 2),
            key=messages.permute(1, 0, 2),
            value=messages.permute(1, 0, 2),
        )[0]
        # [B, AGENT_NUM(NEIGH), 64]
        mes_feats: Tensor = self.mes_mlp(mes_embedding)  # noqa:F841

        # [B, AGENT_NUM(NEIGH), 512]
        agent_output: Tensor = self.param_mlp(agent_embedding)

        ## neighbourhood pointcloud processing
        unet_input = [neigh]  # [B, 3/64/128/256/512, 2L+1, 2L+1, 2L+1]
        for i in range(4):
            conv_feats = self.backbone[i](unet_input[-1])
            unet_input.append(conv_feats)
        unet_output = unet_input[-1]  # [B, 512, 2L+1, 2L+1, 2L+1]

        # [AGENT_NUM(NEIGH), B, 512]
        attn_feats: Tensor = self.neigh_attn(
            query=agent_output.permute(1, 0, 2),
            key=unet_output.flatten(start_dim=2).permute(2, 0, 1),
            value=unet_output.flatten(start_dim=2).permute(2, 0, 1),
        )[0]
        attn_feats += agent_output.permute(1, 0, 2)
        # [AGENT_NUM(NEIGH), B, 512]
        attn_feats: Tensor = (
            self.neigh_ff(attn_feats.view(batch_size * neigh_agent_num, -1)).view(
                neigh_agent_num, batch_size, -1
            )
            + attn_feats
        )

        # [AGENT_NUM(NEIGH), B, 64]
        mlp_feats: Tensor = self.neigh_mlp(
            attn_feats.view(batch_size * neigh_agent_num, -1)
        ).view(neigh_agent_num, batch_size, -1)
        # NOTE(eric): don't know why but should not use += here
        # may raise inplace operation error
        mlp_feats = mlp_feats + mes_embedding.permute(1, 0, 2)

        # out
        implicit_comm: Tensor = self.implicit_comm(  # [B, ACTION_DIM]
            mlp_feats.permute(1, 0, 2)[
                torch.logical_not(self_mask).unsqueeze(-1).repeat(1, 1, 64)
            ].view(-1, neigh_agent_num - 1, 64)
        ).mean(dim=1)
        out: Tensor = self.out(  # [B, ACTION_DIM]
            mlp_feats.permute(1, 0, 2)[self_mask.unsqueeze(-1).repeat(1, 1, 64)].view(
                -1, 64
            )
        )
        # [B, AGENT_NUM(NEIGH), 64]
        explicit_comm: Tensor = self.explicit_comm(mlp_feats.permute(1, 0, 2))

        return out + implicit_comm, explicit_comm


@STRATEGIES.register_module()
class PointNet2Discriminator(Module):
    def __init__(
        self,
        device: str,
        hidden_dim: int,
        cam_types: int,
        attn_head_num: int = 4,
        neigh_out_dim: int = 64,
        min_radius: float = 0.1,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        # [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        action_dim = 10 + self.cam_types

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim, hidden_dim, device=self.device), ReLU()
        )

        # neighbour params
        self.comm_encoder = Linear(param_dim, hidden_dim, device=self.device)
        self.comm_attn = MultiheadAttention(
            embed_dim=hidden_dim, num_heads=attn_head_num, device=self.device
        )
        self.feed_forward = Sequential(
            Linear(hidden_dim, 2 * hidden_dim, device=self.device),
            ReLU(),
            Linear(2 * hidden_dim, hidden_dim, device=self.device),
            ReLU(),
        )

        # neighbourhood
        self.backbone = PointNetBackbone(min_radius)
        self.neigh_merge = Sequential(
            Linear(64 + 256 + 512, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, neigh_out_dim, device=self.device),
            ReLU(),
        )

        # action
        self.action_encoder = Sequential(
            Linear(action_dim, hidden_dim, device=self.device), ReLU()
        )

        # out
        self.out = Sequential(
            Linear(
                3 * hidden_dim + neigh_out_dim,
                hidden_dim,
                device=self.device,
            ),
            ReLU(),
            Linear(hidden_dim, 1, device=self.device),
        )

    def forward(
        self,
        self_and_neigh_params: Tensor,
        self_mask: Tensor,
        neigh: Tensor,
        actions: Tensor,
        voxel_length: float,
    ):
        """
        ## Arguments:

            self_and_neigh_params (Tensor): [B, AGENT_NUM(NEIGH), PARAM_DIM], torch.float.
            Params of agents in `neigh` including current agent itself.

            self_mask (Tensor): [B, AGENT_NUM(NEIGH)], torch.bool. 1 for current agent, 0
            for neighbours.

            neigh (Tensor): [B, 3, 2L+1, 2L+1, 2L+1], torch.float, where `L` is the
            farther dof of the camera. `occupancy`, `install_permitted`,
            `vis_redundancy`. `vis_redundancy` = vis_count / agent_num, -1 if not
            in `must_monitor`.

        ## Returns:

            prob (Tensor): [B, 1], torch.float.
        """

        ## params processing
        batch_size = self_mask.shape[0]
        param_dim = self_and_neigh_params.shape[-1]

        # self
        self_params = self_and_neigh_params[self_mask, :]
        # [B, HIDDEN_DIM]
        self_embedding: Tensor = self.param_encoder(self_params)

        # neighbours
        neigh_params = self_and_neigh_params[torch.logical_not(self_mask), :].view(
            batch_size, -1, param_dim
        )
        # [B, AGENT_NUM(NEIGH), HIDDEN_NUM]
        neigh_agent_embeddings: Tensor = self.comm_encoder(neigh_params)
        # [AGENT_NUM(NEIGH), B, HIDDEN_NUM]
        neigh_agent_seqs: Tensor = neigh_agent_embeddings.permute(1, 0, 2)
        # [1, B, HIDDEN_DIM], ...
        attn_neigh_agent_feats, _ = self.comm_attn(
            query=self_embedding.unsqueeze(0),
            key=neigh_agent_seqs,
            value=neigh_agent_seqs,
        )
        # [B, HIDDEN_DIM]
        comm_feats: Tensor = torch.squeeze(attn_neigh_agent_feats, dim=0)
        comm_feats += self_embedding  # skip connection
        ff_feats = self.feed_forward(comm_feats)
        ff_feats += comm_feats  # skip connection

        ## neighbourhood pointcloud processing
        input_data = _ply2data(neigh, voxel_length)  # [B, N, DATA_DIM]
        _, data = self.backbone(input_data)
        pooling_data: Tensor = [
            torch.max(data[index], dim=1)[0] for index in range(len(data))
        ]
        multi_resolution_data: Tensor = torch.concatenate(  # [B, 64+256+512]
            (pooling_data[1], pooling_data[3], pooling_data[4]), dim=1
        )
        neigh_feats: Tensor = self.neigh_merge(
            multi_resolution_data
        )  # [B, NEIGH_OUT_DIM]

        # action
        action_feats: Tensor = self.action_encoder(actions)  # [B, HIDDEN_DIM]

        # out
        merged_feats: Tensor = torch.concatenate(  # [B, 3*HIDDEN_DIM + NEIGH_OUT_DIM]
            (self_embedding, comm_feats, neigh_feats, action_feats), dim=1
        )
        out: Tensor = sigmoid(self.out(merged_feats))

        return out


def _ply2data(ply_pointcloud: Tensor, voxel_length: float) -> Tensor:
    """
    ## Arguments:

        ply_pointcloud (Tensor): [B, 3, W, D, H], torch.float. `occupancy`,
        `install_permitted`, `vis_redundancy`.
        `vis_redundancy` = vis_count / agent_num, -1 if not in `must_monitor`.

    ## Returns:

        data (Tensor): [B, W*D*H, 9], torch.float, S3DIS format pointcloud.
        [decentralized_x, decentralized_y, decentralized_z, normalized_x,
        normalized_y, normalized_z, normalized_r, normalized_g, normalized_b]
    """
    device = ply_pointcloud.device
    batch_size, _, W, D, H = ply_pointcloud.shape

    pos: Tensor = (  # [B, W*D*H, 3]
        torch.ones_like(ply_pointcloud[0, 0]).nonzero().repeat(batch_size, 1, 1)
        * voxel_length
    )
    center: Tensor = torch.sum(pos, dim=1) / (W * D * H)  # [B, 3]
    decentralized_pos: Tensor = (pos.view(-1, batch_size, 3) - center).view(
        batch_size, -1, 3
    )  # [B, W*D*H, 3]

    max_pos_components: Tensor = torch.max(pos, dim=1)[0]  # [B, 3]
    min_pos_components: Tensor = torch.min(pos, dim=1)[0]  # [B, 3]
    pos_components_distance: Tensor = max_pos_components - min_pos_components  # [B, 3]
    normalized_pos: Tensor = (
        (pos.view(-1, batch_size, 3) - min_pos_components)  # [B, W*D*H, 3]
        / pos_components_distance
    ).view(batch_size, -1, 3)

    color_list: Tensor = torch.tensor(
        [
            [128, 128, 128],  # grey, occupancy
            [255, 255, 0],  # yellow, install_permitted
            [0, 255, 0],  # green, visible
        ],
        dtype=torch.float,
        device=device,
    )
    # [B*W*D*H, 3]
    color: Tensor = torch.zeros((batch_size * W * D * H, 3), device=device)
    occ_index: Tensor = (
        ply_pointcloud[:, 0].contiguous().view(-1).nonzero().view(-1).type(torch.int64)
    )  # [B*W*D*H]
    permit_index: Tensor = (
        ply_pointcloud[:, 1].contiguous().view(-1).nonzero().view(-1).type(torch.int64)
    )  # [B*W*D*H]
    vis_index: Tensor = (
        (ply_pointcloud[:, 2].contiguous().view(-1) > 0)
        .nonzero()
        .view(-1)
        .type(torch.int64)
    )  # [B*W*D*H]
    color[occ_index] = color_list[0]
    color[permit_index] = color_list[1]
    color[vis_index] = ply_pointcloud[:, 2].contiguous().view(-1)[vis_index].view(
        -1, 1
    ) * color_list[2].view(1, -1)
    normalized_color = color.view(batch_size, -1, 3) / 255.0  # [B, W*D*H, 3]

    data: Tensor = torch.concatenate(
        (decentralized_pos, normalized_pos, normalized_color), dim=-1
    )

    return data


def _normalize_param(params: Tensor, room_shape: float) -> Tensor:
    W, D, H = room_shape

    normalized_params = torch.tensor(
        params.cpu().numpy(), dtype=params.dtype, device=params.device
    )

    normalized_params[:, :, [0]] /= W - 1
    normalized_params[:, :, [0]] -= 0.5
    normalized_params[:, :, [1]] /= D - 1
    normalized_params[:, :, [1]] -= 0.5
    normalized_params[:, :, [2]] /= H - 1
    normalized_params[:, :, [2]] -= 0.5

    return normalized_params
