from configs.runtime import DEVICE
from labsurv.utils import get_time_stamp

work_dir = "./output/cart_pole/"
exp_name = get_time_stamp()

episodes = 500
steps = 200

env = dict(
    type="CartPoleEnv",
)

agent_type = "REINFORCE"

if agent_type == "DQN":
    episode_based = False

    agent = dict(
        type="DQN",
        device=DEVICE,
        gamma=0.98,
        explorer_cfg=dict(
            type="BaseEpsilonGreedyExplorer",
            epsilon=0.01,
        ),
        qnet_cfg=dict(
            type="SimpleCNN",
            state_dim=4,
            hidden_dim=128,
            action_dim=2,
            loss_cfg=dict(type="TDLoss"),
        ),
        lr=2e-3,
        to_target_net_interval=5,
        dqn_type="DoubleDQN",
    )

    replay_buffer = dict(
        type="BaseReplayBuffer",
        capacity=10000,
        activate_size=500,
        batch_size=64,
    )
elif agent_type == "REINFORCE":
    episode_based = True

    agent = dict(
        type="REINFORCE",
        device=DEVICE,
        gamma=0.98,
        policy_net_cfg=dict(
            type="SimplePolicyNet",
            state_dim=4,
            hidden_dim=128,
            action_dim=2,
        ),
        lr=2e-3,
    )

logger_cfg = dict(
    type="LoggerHook",
    log_interval=10,
    save_dir=work_dir,
    save_filename=exp_name,
)
