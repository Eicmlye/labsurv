from configs.ocp._base_.params_benchmark import (
    AGENT_NUM, CAM_TYPE_NUM, VOXEL_LENGTH, BACKBONE_PATH, FREEZE_BACKBONE
)
from configs.runtime import DEVICE

expert = dict(
    type="GAIL",
    discriminator_cfg=dict(
        type="PointNet2Discriminator",
        device=DEVICE,
        hidden_dim=64,
        cam_types=CAM_TYPE_NUM,
        comm_attn_head_num=4,
        neigh_out_dim=32,
        voxel_length=VOXEL_LENGTH,
    ),
    device=DEVICE,
    lr=1e-4,
    agent_num=AGENT_NUM,
    gradient_accumulation_batchsize=20,
    backbone_path=BACKBONE_PATH,
    freeze_backbone=FREEZE_BACKBONE,
    expert_data_path="output/expert",
    do_reward_change=False,
    truth_threshold=0.5,
    # load_from=None,
)
