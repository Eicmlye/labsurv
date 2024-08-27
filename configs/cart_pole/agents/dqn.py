from configs.runtime import DEVICE

agent_cfg = dict(
    episode_based = False,

    agent = dict(
        type="DQN",
        device=DEVICE,
        gamma=0.98,
        explorer_cfg=dict(
            type="BaseEpsilonGreedyExplorer",
            epsilon=0.1,
        ),
        qnet_cfg=dict(
            type="SimpleCNN",
            state_dim=4,
            hidden_dim=128,
            action_dim=2,
            loss_cfg=dict(type="TDLoss"),
        ),
        lr=5e-5,
        to_target_net_interval=5,
        dqn_type="DoubleDQN",
        # resume_from="output/cart_pole_trial/episode_9900.pth",
        # load_from="output/cart_pole_trial/episode_9900.pth",
    ),

    replay_buffer = dict(
        type="BaseReplayBuffer",
        device=DEVICE,
        capacity=10000,
        activate_size=500,
        batch_size=200,
        # load_from="output/cart_pole_dqn500_batch2-15/episode_2300.pkl",
    ),
)