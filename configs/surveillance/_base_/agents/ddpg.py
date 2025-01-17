from configs.runtime import DEVICE
from configs.surveillance._base_.envs.loaded_info import SHAPE
from configs.surveillance._base_.envs.std_surveil import cam_intrinsics as CAM_TYPES

pan_section_num = 8
tilt_section_num = 4

agent_cfg = dict(
    is_off_policy=True,
    agent=dict(
        type="OCPDDPG",
        actor_cfg=dict(
            type="OCPDDPGPolicyNet",
            device=DEVICE,
            state_dim=12,
            hidden_dim=256,
            action_dim=3,  # add, del, adjust # , stop
            cam_types=1,
            neck_layers=3,
            pan_section_num=pan_section_num,
            tilt_section_num=tilt_section_num,
        ),
        critic_cfg=dict(
            type="OCPDDPGValueNet",
            device=DEVICE,
            state_dim=12,
            neck_hidden_dim=256,
            adaptive_pooling_dim=10,
            hidden_dim=64,
            neck_layers=2,
            linear_layers=2,
        ),
        explorer_cfg=dict(
            type="OCPEpsilonGreedyExplorer",
            samples=[
                3,
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
        lr=5e-5,
        tau=5e-2,
        # load_from=None,
        # resume_from=None,
    ),
    replay_buffer=dict(
        # type="OCPReplayBuffer",
        type="OCPPrioritizedReplayBuffer",
        device=DEVICE,
        capacity=2000,
        activate_size=500,
        batch_size=200,
        # load_from=None,
    ),
)
