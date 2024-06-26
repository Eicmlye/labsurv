from configs.runtime import DEVICE
from labsurv.utils import get_time_stamp

work_dir = "./output/surveillance/"
exp_name = get_time_stamp()

episodes = 500
steps = 200

env = dict(
    type="BaseSurveillanceEnv",
)

agent = dict(
    type="",
    device=DEVICE,
    gamma=0.98,
    explorer_cfg=dict(
        type="BaseEpsilonGreedyExplorer",
        epsilon=0.01,
    ),
    lr=2e-3,
)

replay_buffer = dict(
    type="BaseReplayBuffer",
    capacity=10000,
    activate_size=500,
    batch_size=64,
)

logger_cfg = dict(
    type="LoggerHook",
    log_interval=10,
    save_dir=work_dir,
    save_filename=exp_name,
)
