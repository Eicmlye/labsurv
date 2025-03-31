from configs.ocp._base_.params_benchmark import (  # noqa: F401
    ALLOW_POLAR,
    BACKBONE_PATH,
    BACKBONE_RADIUS,
    CAM_TYPE_NUM,
    FREEZE_BACKBONE,
    MANUAL,
    PAN_RANGE,
    PAN_SEC_NUM,
    TILT_RANGE,
    TILT_SEC_NUM,
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
            min_radius=BACKBONE_RADIUS,
        ),
        critic_cfg=dict(
            type="PointNet2Critic",
            device=DEVICE,
            hidden_dim=128,
            cam_types=CAM_TYPE_NUM,
            attn_head_num=4,
            env_out_dim=64,
            min_radius=BACKBONE_RADIUS,
        ),
        gradient_accumulation_batchsize=90,
        device=DEVICE,
        gamma=0.95,
        actor_lr=1e-5,
        critic_lr=1e-4,
        update_step=5,
        advantage_param=0.95,
        clip_epsilon=0.1,
        entropy_loss_coef=1e-2,
        pan_section_num=PAN_SEC_NUM,
        tilt_section_num=TILT_SEC_NUM,
        pan_range=PAN_RANGE,
        tilt_range=TILT_RANGE,
        allow_polar=ALLOW_POLAR,
        cam_types=CAM_TYPE_NUM,
        critic_loss_coef=50,
        mixed_reward=True,
        manual=MANUAL,
        backbone_path=BACKBONE_PATH,
        freeze_backbone=FREEZE_BACKBONE,
        # load_from=(
        #     "output/ocp/AGENT_NAME_benchmark/"
        #     "TASK_NAME/"
        #     "WORKING_DIR/"
        #     "models/episode_EPISODE.pth"
        # ),
    ),
)
