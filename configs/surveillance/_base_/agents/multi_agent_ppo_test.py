from configs.runtime import DEVICE
from configs.surveillance._base_.envs.std_surveil_test import (
    cam_intrinsics as CAM_TYPES,
)

cam_type_num = len(CAM_TYPES)

agent_cfg = dict(
    multi_agent=True,
    is_off_policy=False,
    agent=dict(
        type="OCPMultiAgentPPO",
        actor_cfg=dict(
            type="OCPMultiAgentPPOPolicyNet",
            device=DEVICE,
            hidden_dim=128,
            cam_types=cam_type_num,
        ),
        critic_cfg=dict(
            type="OCPMultiAgentPPOValueNet",
            device=DEVICE,
            hidden_dim=128,
        ),
        device=DEVICE,
        gamma=0.98,
        actor_lr=5e-5,
        critic_lr=1e-4,
        update_step=10,
        advantage_param=0.95,
        clip_epsilon=0.5,
        entropy_loss_coef=0.005,
        cam_types=cam_type_num,
        load_from="output/ocp/test_mappo/std_realtime_3w/models/episode_10000.pth",
        # resume_from="output/ocp/ppo/goal_100perc_downtilt/episode_30.pth",
    ),
)
