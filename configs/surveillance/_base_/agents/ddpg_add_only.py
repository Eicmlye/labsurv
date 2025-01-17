from configs.runtime import DEVICE
from configs.surveillance._base_.envs.loaded_info import SHAPE
from configs.surveillance._base_.envs.std_surveil import cam_intrinsics as CAM_TYPES

pan_section_num = 4
tilt_section_num = 3
action_num = 1

agent_cfg = dict(
    is_off_policy=True,
    agent=dict(
        type="OCPDDPG",
        actor_cfg=dict(
            type="OCPDDPGAddOnlyPolicyNet",
            device=DEVICE,
            state_dim=12,
            hidden_dim=64,
            action_dim=action_num,  # add # , del, adjust, stop
            cam_types=1,
            neck_layers=1,
            pan_section_num=pan_section_num,
            tilt_section_num=tilt_section_num,
        ),
        critic_cfg=dict(
            type="OCPDDPGValueNet",
            device=DEVICE,
            state_dim=12,
            neck_hidden_dim=64,
            adaptive_pooling_dim=10,
            hidden_dim=16,
            neck_layers=1,
            linear_layers=1,
        ),
        explorer_cfg=dict(
            type="OCPEpsilonGreedyExplorer",
            samples=[
                action_num,
                SHAPE[0] * SHAPE[1] * SHAPE[2],
                pan_section_num,
                tilt_section_num,
                len(CAM_TYPES),
            ],
            epsilon=0.3,
            pan_section_num=pan_section_num,
            tilt_section_num=tilt_section_num,
        ),
        device=DEVICE,
        gamma=0.98,
        actor_lr=1e-4,
        critic_lr=1e-2,
        tau=2e-1,
        # load_from=None,
        # resume_from="output/ocp/ddpg/trail/episode_20.pth",
    ),
    replay_buffer=dict(
        # type="OCPReplayBuffer",
        type="OCPPrioritizedReplayBuffer",
        device=DEVICE,
        capacity=5000,
        activate_size=100,
        batch_size=80,
        weight=5,
        # load_from="output/ocp/ddpg/trail/replay_buffer.pkl",
    ),
)
