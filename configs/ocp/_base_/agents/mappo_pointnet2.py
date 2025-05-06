from configs.ocp._base_.params import (  # noqa: F401
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
            type="SimpleConvActor",
            device=DEVICE,
            hidden_dim=512,
            cam_types=CAM_TYPE_NUM,
            attn_head_num=8,
            # min_radius=BACKBONE_RADIUS,
        ),
        critic_cfg=dict(
            type="SimpleConvCritic",
            device=DEVICE,
            hidden_dim=512,
            cam_types=CAM_TYPE_NUM,
            attn_head_num=8,
            # min_radius=BACKBONE_RADIUS,
        ),
        gradient_accumulation_batchsize=150,
        device=DEVICE,
        gamma=0.98,
        actor_lr=1e-5,
        critic_lr=1e-4,
        update_step=5,
        advantage_param=0.95,
        clip_epsilon=0.2,
        critic_loss_coef=10,
        entropy_loss_coef=5e-3,
        pan_section_num=PAN_SEC_NUM,
        tilt_section_num=TILT_SEC_NUM,
        pan_range=PAN_RANGE,
        tilt_range=TILT_RANGE,
        allow_polar=ALLOW_POLAR,
        cam_types=CAM_TYPE_NUM,
        mixed_reward=True,
        manual=MANUAL,
        # backbone_path=BACKBONE_PATH,
        freeze_backbone=FREEZE_BACKBONE,
        # load_from=(
        #     "output/ocp/mappo_pointnet2/"
        #     "obs/"
        #     "ma10_30steps_surv_room_test_simpconv_07goal_load1k/"
        #     "models/episode_1000.pth"
        # ),
    ),
)
