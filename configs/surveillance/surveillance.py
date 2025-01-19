from configs.surveillance._base_.agents import ddpg_add_only_clean_agent, ppo_agent
from configs.surveillance._base_.envs import ddpg_add_only_clean_env, ppo_env
from labsurv.utils import get_time_stamp

agent_type = "PPO"

work_dir = f"./output/ocp/{agent_type.lower()}/goal_100perc_downtilt"
exp_name = get_time_stamp()

episodes = 300
steps = 30

agent_cfg = None
if agent_type == "DDPG":
    env = ddpg_add_only_clean_env
    agent_cfg = ddpg_add_only_clean_agent
elif agent_type == "PPO":
    env = ppo_env
    agent_cfg = ppo_agent

env["save_path"] = work_dir
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

eval_interval = save_checkpoint_interval
