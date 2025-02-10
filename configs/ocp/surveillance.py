from configs.ocp._base_.agents import multi_agent_ppo_agent
from configs.ocp._base_.envs import multi_agent_ppo_env
from labsurv.utils import get_time_stamp

agent_type = "MAPPO"

work_dir = f"./output/ocp/{agent_type.lower()}/ma2"
exp_name = get_time_stamp()

episodes = 1
steps = 1

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

if "replay_buffer" in agent_cfg.keys():
    replay_buffer = agent_cfg["replay_buffer"]

save_checkpoint_interval = 1

logger_cfg = dict(
    type="LoggerHook",
    log_interval=1,
    save_dir=work_dir,
    save_filename=exp_name,
)

eval_interval = 1
