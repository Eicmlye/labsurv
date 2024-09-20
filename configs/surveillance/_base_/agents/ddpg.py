from configs.runtime import DEVICE
from configs.surveillance._base_.envs.loaded_info import SHAPE
from configs.surveillance._base_.envs.std_surveil import cam_intrinsics as CAM_TYPES

agent_cfg = dict(
    episode_based=False,
    agent=dict(
        type="OCPDDPG",
        actor_cfg=dict(
            type="OCPDDPGPolicyNet",
            device=DEVICE,
            state_dim=12,
            hidden_dim=256,
            action_dim=4,  # add, del, adjust, stop
            cam_types=1,
            neck_layers=3,
        ),
        critic_cfg=dict(
            type="OCPDDPGValueNet",
            device=DEVICE,
            state_dim=12,
            neck_hidden_dim=256,
            adaptive_pooling_dim=10,
            hidden_dim=64,
            neck_layers=3,
            linear_layers=3,
        ),
        explorer_cfg=dict(
            type="OCPEpsilonGreedyExplorer",
            samples=[4, SHAPE[0] * SHAPE[1] * SHAPE[2], 1.0, 1.0, len(CAM_TYPES)],
            epsilon=0.2,
        ),
        device=DEVICE,
        gamma=0.98,
        lr=5e-5,
        tau=5e-2,
        # load_from=None,
        # resume_from=None,
    ),
    replay_buffer=dict(
        type="OCPReplayBuffer",
        device=DEVICE,
        capacity=1000,
        activate_size=100,
        batch_size=50,
        # load_from=None,
    ),
)
