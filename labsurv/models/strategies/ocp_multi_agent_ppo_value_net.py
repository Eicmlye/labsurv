import torch
from labsurv.builders import STRATEGIES
from torch import Tensor
from torch.nn import Conv3d, Linear, Module, MultiheadAttention, ReLU, Sequential


@STRATEGIES.register_module()
class OCPMultiAgentPPOValueAttentionNet(Module):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        device: str,
        hidden_dim: int,
        conv_out_channels: int,
        cam_types: int,
        attn_head_num: int = 4,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.cam_types = cam_types

        param_dim = 5 + self.cam_types  # [x, y, z, pan, tilt, (one-hot cam_type vec)]

        # neighbourhood
        self.env_conv = Sequential(
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

        # agent params
        self.params_encoder = Sequential(
            Linear(param_dim, hidden_dim, device=self.device), ReLU()
        )

        # cross attention
        self.attn = MultiheadAttention(
            embed_dim=hidden_dim, num_heads=attn_head_num, device=self.device
        )

        # value head
        self.out = Sequential(
            Linear(conv_out_channels + hidden_dim, hidden_dim, device=self.device),
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
        """

        ## neighbourhood
        # [B, CONV_OUT, W, D, H]
        env_feats: Tensor = self.env_conv(env)
        # [W*D*H, B, CONV_OUT]
        env_seqs: Tensor = env_feats.flatten(start_dim=2).permute(2, 0, 1)

        ## agents
        # [B, AGENT_NUM, HIDDEN_DIM]
        agents_embedding: Tensor = self.params_encoder(cam_params)
        # [B, HIDDEN_DIM]
        agents_feat: Tensor = agents_embedding.mean(dim=1)

        ## cross attention
        # [1, B, HIDDEN_DIM], ...
        attn_feats, _ = self.attn(
            query=agents_feat.unsqueeze(0),
            key=env_seqs,
            value=env_seqs,
        )

        ## feat fusion
        # [B, CONV_OUT + HIDDEN_DIM]
        fused_feats: Tensor = torch.cat(
            [torch.squeeze(attn_feats, dim=0), env_seqs.mean(dim=0)], dim=1
        )

        value_predict = self.out(fused_feats)

        return value_predict  # [B, 1]


@STRATEGIES.register_module()
class OCPMultiAgentPPOValueNet(Module):
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

        # neighbourhood
        self.env_conv = Sequential(
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

        # agent params
        self.params_encoder = Sequential(
            Linear(param_dim, hidden_dim, device=self.device), ReLU()
        )

        # value head
        self.out = Sequential(
            Linear(conv_out_channels + hidden_dim, hidden_dim, device=self.device),
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
        """

        ## neighbourhood
        # [B, CONV_OUT, W, D, H]
        env_feats: Tensor = self.env_conv(env)
        # [W*D*H, B, CONV_OUT]
        env_seqs: Tensor = env_feats.flatten(start_dim=2).permute(2, 0, 1)

        ## agents
        # [B, AGENT_NUM, HIDDEN_DIM]
        agents_embedding: Tensor = self.params_encoder(cam_params)
        # [B, HIDDEN_DIM]
        agents_feat: Tensor = agents_embedding.mean(dim=1)

        ## feat fusion
        # [B, CONV_OUT + HIDDEN_DIM]
        fused_feats: Tensor = torch.cat([agents_feat, env_seqs.mean(dim=0)], dim=1)

        value_predict = self.out(fused_feats)

        return value_predict  # [B, 1]
