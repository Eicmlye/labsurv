from configs.runtime import DEVICE
from configs.surveillance._base_.envs.loaded_info import SHAPE
from configs.surveillance._base_.envs.std_surveil import cam_intrinsics as CAM_TYPES

pan_section_num = 8
tilt_section_num = 8
action_num = 1

agent_cfg = dict(
    episode_based=False,
    agent=dict(
        type="OCPDDPG",
        actor_cfg=dict(
            type="OCPDDPGAddOnlyPolicyNet",
            device=DEVICE,
            state_dim=12,
            hidden_dim=128,
            action_dim=action_num,  # add # , del, adjust, stop
            cam_types=1,
            neck_layers=2,
            pan_section_num=pan_section_num,
            tilt_section_num=tilt_section_num,
        ),
        critic_cfg=dict(
            type="OCPDDPGValueNet",
            device=DEVICE,
            state_dim=12,
            neck_hidden_dim=128,
            adaptive_pooling_dim=10,
            hidden_dim=32,
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
            epsilon=0.2,
            pan_section_num=pan_section_num,
            tilt_section_num=tilt_section_num,
            # type="OCPRNDExplorer",
            # device=DEVICE,
            # lr=1e-4,
            # net_cfg=dict(
            #     type="RNDNet",
            #     device=DEVICE,
            #     state_dim=12,
            #     neck_hidden_dim=128,
            #     adaptive_pooling_dim=10,
            #     hidden_dim=32,
            #     neck_layers=1,
            #     linear_layers=1,
            # ),
        ),
        device=DEVICE,
        gamma=0.98,
        lr=1e-3,
        tau=1e-1,
        # load_from=None,
        # resume_from=None,
    ),
    replay_buffer=dict(
        # type="OCPReplayBuffer",
        type="OCPPriorityReplayBuffer",
        device=DEVICE,
        capacity=5000,
        activate_size=100,
        batch_size=50,
        load_from="output/ocp/ddpg/trail/replay_buffer.pkl",
    ),
)
