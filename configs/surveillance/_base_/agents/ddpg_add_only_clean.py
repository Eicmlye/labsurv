from configs.runtime import DEVICE
from configs.surveillance._base_.envs.loaded_info import SHAPE
from configs.surveillance._base_.envs.std_surveil import cam_intrinsics as CAM_TYPES

pan_section_num = 8
tilt_section_num = 8
action_num = 1

agent_cfg = dict(
    episode_based=False,
    agent=dict(
        type="OCPDDPGAddOnlyClean",
        actor_cfg=dict(
            type="OCPDDPGAddOnlyCleanPolicyNet",
            device=DEVICE,
            hidden_dim=32,
            action_dim=action_num,  # add # , del, adjust, stop
            cam_types=1,
            neck_layers=1,
            pan_section_num=pan_section_num,
            tilt_section_num=tilt_section_num,
        ),
        critic_cfg=dict(
            type="OCPDDPGCleanValueNet",
            device=DEVICE,
            neck_hidden_dim=32,
            adaptive_pooling_dim=10,
            neck_layers=1,
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
        actor_lr=1e-5,
        critic_lr=1e-4,
        tau=1e-1,
        # load_from=None,
        # resume_from="output/ocp/ddpg/trail/episode_5.pth",
    ),
    replay_buffer=dict(
        # type="OCPReplayBuffer",
        type="OCPPrioritizedReplayBuffer",
        device=DEVICE,
        capacity=5000,
        activate_size=200,
        batch_size=100,
        weight=4,
        # load_from="output/ocp/ddpg/trail/episode_5.pkl",
        # load_from="output/ocp/ddpg/trail/replay_buffer.pkl",
    ),
)
