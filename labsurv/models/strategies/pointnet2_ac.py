"""
PointNet2 mplementation and checkpoint
from https://github.com/yanx27/Pointnet_Pointnet2_pytorch

Network structure ref: https://github.com/charlesq34/pointnet2
"""

from typing import List

import torch
from labsurv.builders import STRATEGIES
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, MultiheadAttention, ReLU, Sequential

from .pointnet2 import PointNetSetAbstraction


class PointNetBackbone(Module):
    def __init__(self):
        super().__init__()

        self.set_abstraction = ModuleList()

        self.set_abstraction.append(  # [B, 1024, 3], [B, 1024, 64]
            PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        )
        self.set_abstraction.append(  # [B, 256, 3], [B, 256, 128]
            PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        )
        self.set_abstraction.append(  # [B, 64, 3], [B, 64, 256]
            PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        )
        self.set_abstraction.append(  # [B, 16, 3], [B, 16, 512]
            PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
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
        comm_attn_head_num: int = 4,
        neigh_out_dim: int = 64,
        voxel_length: float = 0.2,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        self.voxel_length = voxel_length

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]
        # [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        action_dim = 10 + self.cam_types

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim, hidden_dim, device=self.device), ReLU()
        )

        # neighbour params
        self.comm_encoder = Linear(param_dim, hidden_dim, device=self.device)
        self.comm_attn = MultiheadAttention(
            embed_dim=hidden_dim, num_heads=comm_attn_head_num, device=self.device
        )

        # neighbourhood
        self.backbone = PointNetBackbone()
        self.neigh_merge = Sequential(
            Linear(64 + 256 + 512, neigh_out_dim, device=self.device), ReLU()
        )

        # out
        self.out = Sequential(
            Linear(2 * hidden_dim + neigh_out_dim, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, action_dim, device=self.device),
        )

    def forward(self, self_and_neigh_params: Tensor, self_mask: Tensor, neigh: Tensor):
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

        ## neighbourhood pointcloud processing
        input_data = _ply2data(neigh, self.voxel_length)  # [B, N, DATA_DIM]
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

        # out
        merged_feats: Tensor = torch.concatenate(  # [B, 2*HIDDEN_DIM + NEIGH_OUT_DIM]
            (self_embedding, comm_feats, neigh_feats), dim=1
        )
        out: Tensor = self.out(merged_feats)

        return out


@STRATEGIES.register_module()
class PointNet2Critic(Module):
    def __init__(
        self,
        device: str,
        hidden_dim: int,
        cam_types: int,
        attn_head_num: int = 4,
        env_out_dim: int = 64,
        voxel_length: float = 0.2,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types
        self.voxel_length = voxel_length
        if hidden_dim == 64:
            self.attn_layer = 1
        elif hidden_dim == 128:
            self.attn_layer = 2
        elif hidden_dim == 256:
            self.attn_layer = 3
        elif hidden_dim == 512:
            self.attn_layer = 4
        else:
            raise ValueError(
                f"`hidden_dim` must be in [64, 128, 256, 512], but got {hidden_dim}."
            )

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim, hidden_dim, device=self.device), ReLU()
        )

        # env
        self.backbone = PointNetBackbone()
        self.env_merge = Sequential(
            Linear(64 + 256 + 512, env_out_dim, device=self.device), ReLU()
        )

        # cross attention
        self.attn = MultiheadAttention(
            embed_dim=hidden_dim, num_heads=attn_head_num, device=self.device
        )

        # out
        self.out = Sequential(
            Linear(2 * hidden_dim + env_out_dim, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, 1, device=self.device),
        )

    def forward(self, cam_params: Tensor, env: Tensor) -> Tensor:
        """
        ## Arguments:

            cam_params (Tensor): [B, AGENT_NUM, PARAM_DIM], torch.float. Relative
            coords and relative angles, one-hot cam_type vecs.

            env (Tensor): [B, 3, W, D, H], torch.float, `occupancy`,
            `install_permitted`, `vis_redundancy`.
            `vis_redundancy` = vis_count / agent_num, -1 if not in `must_monitor`.

        ## Returns:

            value_predicted (Tensor): [B, 1], torch.float.
        """
        batch_size = cam_params.shape[0]

        ## env
        input_data: Tensor = _ply2data(env, self.voxel_length)  # [B, N, DATA_DIM]
        _, data = self.backbone(input_data)
        pooling_data: Tensor = [
            torch.max(data[index], dim=1)[0] for index in range(len(data))
        ]
        multi_resolution_pooling_data: Tensor = torch.concatenate(  # [B, 64+256+512]
            (pooling_data[1], pooling_data[3], pooling_data[4]), dim=1
        )
        env_feats: Tensor = self.env_merge(
            multi_resolution_pooling_data
        )  # [B, ENV_OUT_DIM]

        ## agents
        # [B, AGENT_NUM, HIDDEN_DIM]
        agents_embedding: Tensor = self.param_encoder(cam_params)
        # [B, HIDDEN_DIM]
        agents_feats: Tensor = agents_embedding.mean(dim=1)

        ## cross attention
        # [SA_SAMPLE_NUM, B, HIDDEN_DIM]
        env_seqs: Tensor = data[self.attn_layer].permute(1, 0, 2)
        # [1, B, HIDDEN_DIM], ...
        attn_feats, _ = self.attn(
            query=agents_feats.unsqueeze(0),
            key=env_seqs,
            value=env_seqs,
        )

        # out
        merged_feats: Tensor = torch.concatenate(
            (env_feats, agents_feats, attn_feats.view(batch_size, -1)), dim=1
        )
        value_predicted = self.out(merged_feats)

        return value_predicted


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
