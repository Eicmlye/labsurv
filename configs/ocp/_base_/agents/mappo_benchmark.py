from configs.ocp._base_.params_benchmark import (
    AGENT_NUM,
    CAM_TYPE_NUM,
    PAN_RANGE,
    PAN_SEC_NUM,
    TILT_RANGE,
    TILT_SEC_NUM,
    VOXEL_LENGTH,
    ALLOW_POLAR,
)
from configs.runtime import DEVICE

agent_cfg = dict(
    multi_agent=True,
    is_off_policy=False,
    agent=dict(
        type="OCPMultiAgentPPO",
        actor_cfg=dict(
            type="PointNet2Actor",
            device=DEVICE,
            hidden_dim=128,
            cam_types=CAM_TYPE_NUM,
            comm_attn_head_num=4,
            neigh_out_dim=64,
            voxel_length=VOXEL_LENGTH,
        ),
        critic_cfg=dict(
            type="PointNet2Critic",
            device=DEVICE,
            hidden_dim=128,
            cam_types=CAM_TYPE_NUM,
            attn_head_num=4,
            env_out_dim=64,
            voxel_length=VOXEL_LENGTH,
        ),
        device=DEVICE,
        gamma=0.9,
        gradient_accumulation_batchsize=40,
        actor_lr=1e-5,
        critic_lr=1e-4,
        update_step=5,
        advantage_param=0.95,
        clip_epsilon=0.01,
        entropy_loss_coef=3,
        agent_num=AGENT_NUM,
        pan_section_num=PAN_SEC_NUM,
        tilt_section_num=TILT_SEC_NUM,
        pan_range=PAN_RANGE,
        tilt_range=TILT_RANGE,
        allow_polar=ALLOW_POLAR,
        cam_types=CAM_TYPE_NUM,
        # mixed_reward=True,
        backbone_path="labsurv/checkpoint/pointnet2backbone_rename.pth",
        freeze_backbone=[0, 1, 2, 3],
        # load_from="output/ocp/mappo_pointnet2/ma10_30steps_80to85_load300/models/episode_700.pth",
        # resume_from="output/ocp/AGENT_NAME/EXP_NAME/models/episode_EPI_NUM.pth",
    ),
)
