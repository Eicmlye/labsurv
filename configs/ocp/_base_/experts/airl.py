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
        type="PointNet2Approx",
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
    appr_lr=1e-3,
    shaping_lr=1e-3,
    agent_num=AGENT_NUM,
    gradient_accumulation_batchsize=25,
    backbone_path=BACKBONE_PATH,
    freeze_backbone=FREEZE_BACKBONE,
    expert_data_path="output/expert",
    do_reward_change=False,
    truth_threshold=0.3,
    # load_from=(
    #     "output/ocp/mappo_pointnet2_benchmark/"
    #     "ma5_AC_03_airl_melt/"
    #     "ma5_20steps_AC_03_32melt_airl_keepreward/"
    #     "imitators/episode_100.pth"
    # ),
)
