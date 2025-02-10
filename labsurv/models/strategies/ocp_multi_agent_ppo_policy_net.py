import torch
import torch.nn.functional as F
from labsurv.builders import STRATEGIES
from torch import Tensor
from torch.nn import Conv3d, Linear, Module, MultiheadAttention, ReLU, Sequential


@STRATEGIES.register_module()
class OCPMultiAgentPPOPolicyAttentionNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: str,
        hidden_dim: int,
        cam_types: int,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]
        conv_out_channels = 16
        space_attn_head_num = 1
        comm_attn_head_num = 1
        # [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        action_dim = 10 + self.cam_types

        # neighbourhood
        self.local_env_conv = Sequential(
            Conv3d(
                in_channels=3,
                out_channels=conv_out_channels // 2,
                kernel_size=3,
                padding=1,
                device=self.device,
            ),
            ReLU(),
            Conv3d(
                in_channels=conv_out_channels // 2,
                out_channels=conv_out_channels,
                kernel_size=3,
                padding=1,
                device=self.device,
            ),
            ReLU(),
        )
        self.space_attn = MultiheadAttention(
            embed_dim=conv_out_channels,
            num_heads=space_attn_head_num,
            device=self.device,
        )

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim, hidden_dim, device=self.device), ReLU()
        )

        # neighbour params
        self.comm_encoder = Linear(param_dim, hidden_dim, device=self.device)
        self.comm_attn = MultiheadAttention(
            embed_dim=hidden_dim, num_heads=comm_attn_head_num, device=self.device
        )

        # action head
        self.out = Sequential(
            Linear(conv_out_channels + hidden_dim * 2, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, action_dim, device=self.device),
        )

    def forward(
        self, self_and_neigh_params: Tensor, self_mask: Tensor, neigh: Tensor
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
        batch_size = self_mask.shape[0]
        param_dim = self_and_neigh_params.shape[-1]

        ## neighbourhood
        # [B, CONV_OUT, 2L+1, 2L+1, 2L+1]
        neigh_feats: Tensor = self.local_env_conv(neigh)
        # [(2L+1)^3, B, CONV_OUT]
        neigh_seqs: Tensor = neigh_feats.flatten(start_dim=2).permute(2, 0, 1)
        # [(2L+1)^3, B, CONV_OUT], ...
        attn_neigh_feats, _ = self.space_attn(neigh_seqs, neigh_seqs, neigh_seqs)
        # pooling, [B, CONV_OUT]
        local_feats = torch.mean(attn_neigh_feats, dim=0)

        ## self
        self_params = self_and_neigh_params[self_mask, :]
        # [B, HIDDEN_DIM]
        self_embedding: Tensor = self.param_encoder(self_params)

        ## neighbours
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

        ## feat fusion
        # [B, CONV_OUT + 2 * HIDDEN_DIM]
        fused_feats: Tensor = torch.cat(
            [local_feats, self_embedding, comm_feats], dim=1
        )

        ## action dist
        dist: Tensor = F.softmax(self.out(fused_feats), dim=-1)

        return dist  # [B, ACTION_DIM]


@STRATEGIES.register_module()
class OCPMultiAgentPPOPolicyComplexNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: str,
        hidden_dim: int,
        conv_out_channels: int,
        cam_types: int,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]
        # [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        action_dim = 10 + self.cam_types

        # neighbourhood
        self.local_env_conv = Sequential(
            Conv3d(
                in_channels=3,
                out_channels=conv_out_channels // 2,
                kernel_size=3,
                padding=1,
                device=self.device,
            ),
            ReLU(),
            Conv3d(
                in_channels=conv_out_channels // 2,
                out_channels=conv_out_channels,
                kernel_size=3,
                padding=1,
                device=self.device,
            ),
            ReLU(),
        )

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim, hidden_dim, device=self.device), ReLU()
        )

        # neighbour params
        self.comm_encoder = Linear(param_dim, hidden_dim, device=self.device)

        # action head
        self.out = Sequential(
            Linear(conv_out_channels + hidden_dim * 2, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, action_dim, device=self.device),
        )

    def forward(
        self, self_and_neigh_params: Tensor, self_mask: Tensor, neigh: Tensor
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
        batch_size = self_mask.shape[0]
        param_dim = self_and_neigh_params.shape[-1]

        ## neighbourhood
        # [B, CONV_OUT, 2L+1, 2L+1, 2L+1]
        neigh_feats: Tensor = self.local_env_conv(neigh)
        # [(2L+1)^3, B, CONV_OUT]
        neigh_seqs: Tensor = neigh_feats.flatten(start_dim=2).permute(2, 0, 1)
        # pooling, [B, CONV_OUT]
        local_feats = torch.mean(neigh_seqs, dim=0)

        ## self
        self_params = self_and_neigh_params[self_mask, :]
        # [B, HIDDEN_DIM]
        self_embedding: Tensor = self.param_encoder(self_params)

        ## neighbours
        neigh_params = self_and_neigh_params[torch.logical_not(self_mask), :].view(
            batch_size, -1, param_dim
        )
        # [B, AGENT_NUM(NEIGH), HIDDEN_NUM]
        neigh_agent_embeddings: Tensor = self.comm_encoder(neigh_params)
        # [AGENT_NUM(NEIGH), B, HIDDEN_NUM]
        neigh_agent_seqs: Tensor = neigh_agent_embeddings.permute(1, 0, 2)
        # [B, HIDDEN_DIM]
        neigh_agent_feats: Tensor = torch.mean(neigh_agent_seqs, dim=0)

        ## feat fusion
        # [B, CONV_OUT + 2 * HIDDEN_DIM]
        fused_feats: Tensor = torch.cat(
            [local_feats, self_embedding, neigh_agent_feats], dim=1
        )

        ## action dist
        dist: Tensor = F.softmax(self.out(fused_feats), dim=-1)

        return dist  # [B, ACTION_DIM]


@STRATEGIES.register_module()
class OCPMultiAgentPPOPolicyNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: str,
        hidden_dim: int,
        conv_out_channels: int,
        cam_types: int,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]
        # [-x, +x, -y, +y, -z, +z, -p, +p, -t, +t, (change to cam_type n)]
        action_dim = 10 + self.cam_types

        # neighbourhood
        self.local_env_conv = Sequential(
            Conv3d(
                in_channels=3,
                out_channels=conv_out_channels // 2,
                kernel_size=3,
                padding=1,
                device=self.device,
            ),
            ReLU(),
            Conv3d(
                in_channels=conv_out_channels // 2,
                out_channels=conv_out_channels,
                kernel_size=3,
                padding=1,
                device=self.device,
            ),
            ReLU(),
        )

        # self param
        self.param_encoder = Sequential(
            Linear(param_dim, hidden_dim, device=self.device), ReLU()
        )

        # neighbour params
        self.comm_encoder = Linear(param_dim, hidden_dim, device=self.device)

        # action head
        self.out = Sequential(
            Linear(conv_out_channels + hidden_dim * 2, hidden_dim, device=self.device),
            ReLU(),
            Linear(hidden_dim, action_dim, device=self.device),
        )

    def forward(
        self, self_and_neigh_params: Tensor, self_mask: Tensor, neigh: Tensor
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

        batch_size = self_mask.shape[0]
        param_dim = self_and_neigh_params.shape[-1]

        ## neighbourhood
        # [B, CONV_OUT, 2L+1, 2L+1, 2L+1]
        neigh_feats: Tensor = self.local_env_conv(neigh)
        # [(2L+1)^3, B, CONV_OUT]
        neigh_seqs: Tensor = neigh_feats.flatten(start_dim=2).permute(2, 0, 1)
        # pooling, [B, CONV_OUT]
        local_feats = torch.mean(neigh_seqs, dim=0)

        ## self
        self_params = self_and_neigh_params[self_mask, :]
        # [B, HIDDEN_DIM]
        self_embedding: Tensor = self.param_encoder(self_params)

        ## neighbours
        neigh_params = self_and_neigh_params[torch.logical_not(self_mask), :].view(
            batch_size, -1, param_dim
        )
        # [B, AGENT_NUM(NEIGH), HIDDEN_NUM]
        neigh_agent_embeddings: Tensor = self.comm_encoder(neigh_params)
        # [AGENT_NUM(NEIGH), B, HIDDEN_NUM]
        neigh_agent_seqs: Tensor = neigh_agent_embeddings.permute(1, 0, 2)
        # [B, HIDDEN_DIM]
        neigh_agent_feats: Tensor = torch.mean(neigh_agent_seqs, dim=0)

        ## feat fusion
        # [B, CONV_OUT + 2 * HIDDEN_DIM]
        fused_feats: Tensor = torch.cat(
            [local_feats, self_embedding, neigh_agent_feats], dim=1
        )

        ## action dist
        logits: Tensor = self.out(fused_feats)

        return logits  # [B, ACTION_DIM]
