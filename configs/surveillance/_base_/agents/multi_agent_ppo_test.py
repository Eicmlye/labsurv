from configs.runtime import DEVICE
from configs.surveillance._base_.params import (
    AGENT_NUM,
    CAM_TYPE_NUM,
    PAN_RANGE,
    PAN_SEC_NUM,
    TILT_RANGE,
    TILT_SEC_NUM,
)

agent_cfg = dict(
    multi_agent=True,
    is_off_policy=False,
    agent=dict(
        type="OCPMultiAgentPPO",
        actor_cfg=dict(
            type="OCPMultiAgentPPOPolicyNet",
            device=DEVICE,
            hidden_dim=128,
            conv_out_channels=64,
            cam_types=CAM_TYPE_NUM,
        ),
        critic_cfg=dict(
            type="OCPMultiAgentPPOValueNet",
            device=DEVICE,
            hidden_dim=128,
            conv_out_channels=64,
            cam_types=CAM_TYPE_NUM,
        ),
        device=DEVICE,
        gamma=0.98,
        actor_lr=1e-5,
        critic_lr=1e-4,
        update_step=5,
        advantage_param=0.95,
        clip_epsilon=0.05,
        entropy_loss_coef=2,
        agent_num=AGENT_NUM,
        pan_section_num=PAN_SEC_NUM,
        tilt_section_num=TILT_SEC_NUM,
        pan_range=PAN_RANGE,
        tilt_range=TILT_RANGE,
        cam_types=CAM_TYPE_NUM,
        mixed_reward=False,
        load_from="output/ocp/test_mappo/ma6_realtime_70from50_rewarddelta/models/episode_400.pth",
        # resume_from="output/ocp/ppo/goal_100perc_downtilt/episode_30.pth",
    ),
)
