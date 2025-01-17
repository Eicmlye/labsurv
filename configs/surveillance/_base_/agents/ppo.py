from configs.runtime import DEVICE
from configs.surveillance._base_.envs.loaded_info import SHAPE
from configs.surveillance._base_.envs.std_surveil import cam_intrinsics as CAM_TYPES

action_num = 1  # add
pan_section_num = 8
tilt_section_num = 8
cam_type_num = len(CAM_TYPES)

agent_cfg = dict(
    is_off_policy=False,
    agent=dict(
        type="OCPPPO",
        actor_cfg=dict(
            type="OCPPPOPolicyNet",
            device=DEVICE,
            hidden_dim=32,
            action_dim=action_num,
            cam_types=cam_type_num,
            neck_layers=2,
            pan_section_num=pan_section_num,
            tilt_section_num=tilt_section_num,
            adaptive_pooling_dim=10,
        ),
        critic_cfg=dict(
            type="OCPPPOValueNet",
            device=DEVICE,
            neck_hidden_dim=32,
            adaptive_pooling_dim=10,
            neck_layers=2,
        ),
        explorer_cfg=dict(
            type="OCPEpsilonGreedyExplorer",
            action_num=action_num,
            room_shape=SHAPE,
            cam_types=cam_type_num,
            epsilon=0.2,
            pan_section_num=pan_section_num,
            tilt_section_num=tilt_section_num,
        ),
        device=DEVICE,
        gamma=0.98,
        actor_lr=1e-6,
        critic_lr=5e-5,
        update_step=10,
        advantage_param=0.95,
        clip_epsilon=0.2,
        pan_section_num=pan_section_num,
        tilt_section_num=tilt_section_num,
        # load_from="output/ocp/ppo/trail/episode_35.pth",
        resume_from="output/ocp/ppo/trail/episode_10.pth",
    ),
)
