from configs.cart_pole.agents import dqn_agent, reinforce_agent
from labsurv.utils import get_time_stamp

work_dir = "./output/cart_pole/"
exp_name = get_time_stamp()

episodes = 20000
steps = 500000

env = dict(
    type="CartPoleEnv",
)

agent_type = "REINFORCE"

if agent_type == "DQN":
    agent_cfg = dqn_agent
elif agent_type == "REINFORCE":
    agent_cfg = reinforce_agent

episode_based = agent_cfg["episode_based"]
agent = agent_cfg["agent"]
if "replay_buffer" in agent_cfg.keys():
    replay_buffer = agent_cfg["replay_buffer"]

logger_cfg = dict(
    type="LoggerHook",
    log_interval=10,
    save_dir=work_dir,
    save_filename=exp_name,
)

save_checkpoint_interval = 1000
