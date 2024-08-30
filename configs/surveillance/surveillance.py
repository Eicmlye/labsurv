from configs.surveillance._base_.agents import reinforce_agent
from configs.surveillance._base_.envs import room_data_path
from labsurv.utils import get_time_stamp

work_dir = "./output/surveillance/"
exp_name = get_time_stamp()

episodes = 2000
steps = 10

env = dict(
    type="BaseSurveillanceEnv",
    room_data_path=room_data_path,
)

agent_type = "REINFORCE"

if agent_type == "REINFORCE":
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

save_checkpoint_interval = 100
