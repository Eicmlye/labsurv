from configs.ocp._base_.agents import mappo_pointnet2_agent
from configs.ocp._base_.envs import mappo_pointnet2_env
from configs.ocp._base_.experts import gail_expert  # noqa: F401
from configs.ocp._base_.params import AGENT_NUM, MANUAL, ROOM_NAME
from labsurv.utils import get_time_stamp

agent_type = "MAPPO_PointNet2"

episodes = 500
steps = 30

task_name = "obs_test"
task_conditions = "01r_invrew_xmelt"

work_dir = (
    f"./output/ocp/{agent_type.lower()}/"
    + f"{task_name}/"
    + f"ma{AGENT_NUM}_{steps}steps_{ROOM_NAME}"
    + f"_{task_conditions}"
    + ("_MANUAL" if MANUAL else "")
)
exp_name = get_time_stamp()

agent_cfg = None
if agent_type == "MAPPO_PointNet2":
    env = mappo_pointnet2_env
    agent_cfg = mappo_pointnet2_agent

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
