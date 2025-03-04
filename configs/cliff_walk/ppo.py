from configs.runtime import DEVICE
from labsurv.utils import get_time_stamp

agent_type = "PPO"

work_dir = f"./output/cliff_walk/{agent_type.lower()}/10x11"
exp_name = get_time_stamp()

episodes = 10000
steps = 100

FREE = 0
FROM = 1
DEAD = 2
DEST = 3

cliff_env = [
    [FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE],
    [FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE],
    [FREE, FREE, DEAD, FREE, DEAD, DEAD, DEAD, DEAD, DEAD, FREE, FREE],
    [FREE, FREE, DEAD, FREE, DEAD, FREE, FREE, FREE, DEAD, FREE, FREE],
    [FREE, FREE, DEAD, FREE, DEAD, FREE, DEAD, FREE, DEAD, FREE, FREE],
    [FREE, FREE, DEAD, FREE, DEAD, DEST, FREE, FREE, DEAD, FREE, FREE],
    [FREE, FREE, DEAD, FREE, DEAD, DEAD, DEAD, FREE, DEAD, FREE, FREE],
    [FREE, FREE, DEAD, FREE, FREE, FREE, FREE, FREE, DEAD, FREE, FREE],
    [FREE, FREE, DEAD, FREE, FREE, FREE, FREE, FREE, DEAD, FREE, FREE],
    [FROM, FREE, DEAD, DEAD, DEAD, DEAD, DEAD, DEAD, DEAD, FREE, DEST],
    # ========
    # [FREE, FREE, FREE, FREE, FREE, FREE],
    # [FREE, FREE, FREE, FREE, FREE, FREE],
    # [FREE, FREE, FREE, FREE, FREE, FREE],
    # [FROM, FREE, DEAD, DEAD, FREE, DEST],
    # ========
    # [FREE, FREE, FREE],
    # [FREE, FREE, FREE],
    # [FROM, DEAD, DEST],
]

agent_cfg = None
if agent_type == "PPO":
    env = dict(
        type="CliffWalkActorCriticEnv",
        env=cliff_env,
        device=DEVICE,
    )
    agent_cfg = dict(
        type="CliffWalkPPO",
        device=DEVICE,
        gamma=0.98,
        actor_cfg=dict(
            device=DEVICE,
            state_dim=len(cliff_env) * len(cliff_env[0]),
            hidden_dim=128,
            action_dim=4,
        ),
        critic_cfg=dict(
            device=DEVICE,
            state_dim=len(cliff_env) * len(cliff_env[0]),
            hidden_dim=128,
        ),
        actor_lr=1e-4,
        critic_lr=1e-3,
        update_step=10,
        entropy_loss_coef=0.01,
        advantage_coef=0.95,
        clip_epsilon=0.1,
        # load_from="output/cliff_walk/ac/3x3_3k/episode_3000.pth"
    )

env["save_path"] = work_dir
agent = agent_cfg

runner = dict(
    type="CliffWalkActorCriticRunner",
)

save_checkpoint_interval = 1000

logger_cfg = dict(
    type="LoggerHook",
    log_interval=100,
    save_dir=work_dir,
    save_filename=exp_name,
)

eval_interval = 200
