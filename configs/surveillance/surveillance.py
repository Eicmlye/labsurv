from configs.runtime import DEVICE
from configs.surveillance._base_.agents import (
    ddpg_add_only_agent,
    ddpg_add_only_clean_agent,
    ddpg_agent,
    reinforce_agent,
)
from labsurv.utils import get_time_stamp

agent_type = "DDPG"
# action_type = "AddOnly"
action_type = "AddOnlyClean"

work_dir = f"./output/ocp/{agent_type.lower()}/trail"
exp_name = get_time_stamp()

episodes = 1000
steps = 10

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
    env["type"] = "OCPDDPG" + action_type + "Env"
    if action_type == "":
        agent_cfg = ddpg_agent
    elif action_type == "AddOnly":
        agent_cfg = ddpg_add_only_agent
    elif action_type == "AddOnlyClean":
        agent_cfg = ddpg_add_only_clean_agent
    else:
        raise NotImplementedError(f"Unknown action strategy \"{action_type}\"")
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
    log_interval=5,
    save_dir=work_dir,
    save_filename=exp_name,
)

save_checkpoint_interval = 5

eval_interval = 20
