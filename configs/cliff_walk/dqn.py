from configs.runtime import DEVICE
from labsurv.utils import get_time_stamp

agent_type = "DQN"

work_dir = f"./output/cliff_walk/{agent_type.lower()}/4x6"
exp_name = get_time_stamp()

episodes = 1000
steps = 200

FREE = 0
FROM = 1
DEAD = 2
DEST = 3

cliff_env = [
    [FREE, FREE, FREE, FREE, FREE, FREE],
    [FREE, FREE, FREE, FREE, FREE, FREE],
    [FREE, FREE, FREE, FREE, FREE, FREE],
    [FROM, FREE, DEAD, DEAD, FREE, DEST],
    # ========
    # [FREE, FREE, FREE],
    # [FREE, FREE, FREE],
    # [FROM, DEAD, DEST],
]

agent_cfg = None
if agent_type == "DQN":
    env = dict(
        type="CliffWalkDQNEnv",
        env=cliff_env,
        device=DEVICE,
    )
    agent_cfg = dict(
        type="CliffWalkDQN",
        device=DEVICE,
        gamma=0.98,
        qnet_cfg=dict(
            device=DEVICE,
            state_dim=len(cliff_env) * len(cliff_env[0]),
            hidden_dim=128,
            action_dim=4,
        ),
        lr=1e-3,
        explorer_cfg=dict(
            type="BaseEpsilonGreedyExplorer",
            samples=4,
            epsilon=0.999,
            epsilon_decay=0.994,
            epsilon_min=0.05,
        ),
    )

env["save_path"] = work_dir
agent = agent_cfg

runner = dict(
    type="CliffWalkDQNRunner",
)

replay_buffer = dict(
    type="CliffWalkPrioritizedReplayBuffer",
    device=DEVICE,
    batch_size=100,
    capacity=5000,
    activate_size=500,
    weight=6,
)

save_checkpoint_interval = 20

logger_cfg = dict(
    type="LoggerHook",
    log_interval=5,
    save_dir=work_dir,
    save_filename=exp_name,
)

eval_interval = 5
