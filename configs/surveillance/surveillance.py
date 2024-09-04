from configs.runtime import DEVICE
from configs.surveillance._base_.agents import reinforce_agent
from labsurv.utils import get_time_stamp

work_dir = "./output/surveillance/"
exp_name = get_time_stamp()

runner = dict(
    type="OCPEpisodeBasedRunner",
)

episodes = 10000
steps = 50

env = dict(
    type="BaseSurveillanceEnv",
    room_data_path="output/surv_room/SurveillanceRoom.pkl",
    device=DEVICE,
    save_path=work_dir,
)

agent_type = "REINFORCE"
agent_cfg = None
if agent_type == "REINFORCE":
    agent_cfg = reinforce_agent
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
