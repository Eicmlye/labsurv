from configs.ocp._base_.params_benchmark import (  # noqa: F401
    AGENT_NUM,
    BACKBONE_PATH,
    BACKBONE_RADIUS,
    CAM_TYPE_NUM,
    FREEZE_BACKBONE,
)
from configs.runtime import DEVICE

expert = dict(
    type="GAIL",
    discriminator_cfg=dict(
        type="PointNet2Discriminator",
        device=DEVICE,
        hidden_dim=128,
        cam_types=CAM_TYPE_NUM,
        comm_attn_head_num=4,
        neigh_out_dim=64,
        min_radius=BACKBONE_RADIUS,
    ),
    device=DEVICE,
    lr=1e-4,
    agent_num=AGENT_NUM,
    gradient_accumulation_batchsize=45,
    backbone_path=BACKBONE_PATH,
    freeze_backbone=FREEZE_BACKBONE,
    expert_data_path="/root/autodl-tmp/output/expert",
    do_reward_change=True,
    truth_threshold=0.3,
    disc_reward_base=0.5,
    load_from=(
        "/root/autodl-tmp/output/ocp/mappo_pointnet2_test/"
        "AC_03/"
        "ma3_30steps_AC_03_gail_32melt/"
        "imitators/episode_500.pth"
    ),
)
