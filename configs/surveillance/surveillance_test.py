from configs.surveillance._base_.agents.multi_agent_ppo_test import (
    agent_cfg as multi_agent_ppo_agent,
)
from configs.surveillance._base_.envs.multi_agent_ppo_test_env import (
    env_cfg as multi_agent_ppo_env,
)
from labsurv.utils import get_time_stamp

agent_type = "MAPPO"

work_dir = f"./output/ocp/test_{agent_type.lower()}/debug"
exp_name = get_time_stamp()

episodes = 10000
steps = 50

agent_cfg = None
if agent_type == "MAPPO":
    env = multi_agent_ppo_env
    agent_cfg = multi_agent_ppo_agent

env["save_path"] = work_dir
agent = agent_cfg["agent"]

runner = dict(type=None)
if agent_cfg["multi_agent"]:
    if not agent_cfg["is_off_policy"]:
        runner["type"] = "OCPMultiAgentOnPolicyRunner"
else:
    if not agent_cfg["is_off_policy"]:
        runner["type"] = "OCPOnPolicyRunner"
    else:
        runner["type"] = "OCPOffPolicyRunner"

if "replay_buffer" in agent_cfg.keys():
    replay_buffer = agent_cfg["replay_buffer"]

save_checkpoint_interval = 100

logger_cfg = dict(
    type="LoggerHook",
    log_interval=100,
    save_dir=work_dir,
    save_filename=exp_name,
)

eval_interval = 20
