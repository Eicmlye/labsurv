from configs.ocp._base_.agents import mappo_benchmark_agent
from configs.ocp._base_.envs import mappo_benchmark_env
from configs.ocp._base_.experts import gail_expert  # noqa: F401
from configs.ocp._base_.params_benchmark import AGENT_NUM, BENCHMARK_NAME, MANUAL
from labsurv.utils import get_time_stamp

agent_type = "MAPPO_PointNet2"

episodes = 1000
steps = 30

task_name = "AC_03"
task_conditions = "invrew_xmelt"

work_dir = (
    f"./output/ocp/{agent_type.lower()}_benchmark/"
    + f"{task_name}/"
    + f"ma{AGENT_NUM}_{steps}steps_{BENCHMARK_NAME}"
    + f"_{task_conditions}"
    + ("_MANUAL" if MANUAL else "")
)
exp_name = get_time_stamp()

agent_cfg = None
if agent_type == "MAPPO_PointNet2":
    env = mappo_benchmark_env
    agent_cfg = mappo_benchmark_agent

env["save_path"] = work_dir
agent = agent_cfg["agent"]

runner = dict(type=None)
if agent_cfg["multi_agent"]:
    if not agent_cfg["is_off_policy"]:
        runner["type"] = "OCPMultiAgentOnPolicyRunner"

if "replay_buffer" in agent_cfg.keys():
    replay_buffer = agent_cfg["replay_buffer"]

# expert = gail_expert

save_checkpoint_interval = 100

logger_cfg = dict(
    type="LoggerHook",
    log_interval=5,
    save_dir=work_dir,
    save_filename=exp_name,
)

eval_interval = 5
