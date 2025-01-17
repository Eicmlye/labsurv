from configs.runtime import DEVICE
from configs.surveillance._base_.agents import (
    ddpg_add_only_agent,
    ddpg_add_only_clean_agent,
    ddpg_agent,
    ppo_agent,
    reinforce_agent,
)
from labsurv.utils import get_time_stamp

agent_type = "PPO"
action_type = None
# action_type = "AddOnly"
# action_type = "AddOnlyClean"

work_dir = f"./output/ocp/{agent_type.lower()}/trail"
exp_name = get_time_stamp()

episodes = 1000
steps = 75

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
elif agent_type == "PPO":
    env["type"] = "OCPPPOEnv"
    agent_cfg = ppo_agent
agent = agent_cfg["agent"]

runner = dict(
    type=("OCPOffPolicyRunner" if agent_cfg["is_off_policy"] else "OCPOnPolicyRunner"),
)

if "replay_buffer" in agent_cfg.keys():
    replay_buffer = agent_cfg["replay_buffer"]

save_checkpoint_interval = 5

logger_cfg = dict(
    type="LoggerHook",
    log_interval=save_checkpoint_interval,
    save_dir=work_dir,
    save_filename=exp_name,
)

eval_interval = 20
