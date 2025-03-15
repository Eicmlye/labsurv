from configs.ocp._base_.params_benchmark import (  # noqa: F401
    AGENT_NUM,
    BACKBONE_PATH,
    CAM_TYPE_NUM,
    FREEZE_BACKBONE,
    VOXEL_LENGTH,
)
from configs.runtime import DEVICE

expert = dict(
    type="AIRL",
    reward_approximator_cfg=dict(
        type="PointNet2Discriminator",
        device=DEVICE,
        hidden_dim=128,
        cam_types=CAM_TYPE_NUM,
        comm_attn_head_num=4,
        neigh_out_dim=64,
        voxel_length=VOXEL_LENGTH,
    ),
    reward_shaping_cfg=dict(
        type="PointNet2Shaping",
        device=DEVICE,
        hidden_dim=128,
        cam_types=CAM_TYPE_NUM,
        comm_attn_head_num=4,
        neigh_out_dim=64,
        voxel_length=VOXEL_LENGTH,
    ),
    device=DEVICE,
    appr_lr=5e-4,
    shaping_lr=5e-4,
    agent_num=AGENT_NUM,
    gradient_accumulation_batchsize=20,
    backbone_path=BACKBONE_PATH,
    freeze_backbone=FREEZE_BACKBONE,
    expert_data_path="output/expert",
    do_reward_change=True,
    truth_threshold=0.3,
    # load_from="output/ocp/mappo_pointnet2_benchmark/ma5_AC_03_gail_task/ma5_20steps_AC_03_32melt_load1k_gail_32melt_keepreward_expertencodeaction_load200/imitators/episode_1000.pth",
)
