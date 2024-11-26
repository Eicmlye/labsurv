from configs.runtime import DEVICE
from configs.surveillance._base_.agents import ddpg_agent, reinforce_agent
from labsurv.utils import get_time_stamp

agent_type = "DDPG"

work_dir = f"./output/ocp/{agent_type.lower()}/trail"
exp_name = get_time_stamp()

episodes = 10000
steps = 30

env = dict(
    type=None,
    room_data_path="output/surv_room/SurveillanceRoom.pkl",
    device=DEVICE,
    save_path=work_dir,
)

agent_cfg = None
if agent_type == "REINFORCE":
    env["type"] = "OCPREINFORCEEnv"
    agent_cfg = reinforce_agent
elif agent_type == "DDPG":
    env["type"] = "OCPDDPGEnv"
    agent_cfg = ddpg_agent
agent = agent_cfg["agent"]

runner = dict(
    type=(
        "OCPEpisodeBasedRunner" if agent_cfg["episode_based"] else "OCPStepBasedRunner"
    ),
)

if "replay_buffer" in agent_cfg.keys():
    replay_buffer = agent_cfg["replay_buffer"]

logger_cfg = dict(
    type="LoggerHook",
    log_interval=10,
    save_dir=work_dir,
    save_filename=exp_name,
)

save_checkpoint_interval = 100

eval_interval = 20
