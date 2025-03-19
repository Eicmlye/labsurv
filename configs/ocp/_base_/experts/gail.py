from configs.ocp._base_.params_benchmark import (  # noqa: F401
    AGENT_NUM,
    BACKBONE_PATH,
    BACKBONE_RADIUS,
    CAM_TYPE_NUM,
    FREEZE_BACKBONE,
    VOXEL_LENGTH,
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
        voxel_length=VOXEL_LENGTH,
        min_radius=BACKBONE_RADIUS,
    ),
    device=DEVICE,
    lr=5e-4,
    agent_num=AGENT_NUM,
    gradient_accumulation_batchsize=25,
    backbone_path=BACKBONE_PATH,
    freeze_backbone=FREEZE_BACKBONE,
    expert_data_path="output/expert",
    do_reward_change=False,
    truth_threshold=0.3,
    # load_from="output/ocp/AGENT_NAME_benchmark/TASK_NAME/WORKING_DIR/imitators/episode_EPISODE.pth",
)
