import os.path as osp

from configs.runtime import DEVICE
from labsurv.utils import get_time_stamp

work_dir = "./output/cart_pole/"
exp_name = "cart_pole" + "_" + get_time_stamp()

episodes = 500
batch_size = 64

env = dict(
  id="CartPole-v0",
  type="",
)

agent = dict(
  type="DQN",
  qnet_cfg=dict(
    type="SimpleQNet",
    state_dim=4,
    hidden_dim=128,
    action_dim=2,
    loss_cfg=dict(type="TDLoss")
  ),
  lr=2e-3,
  gamma=0.98,
  greedy_epsilon=0.01,
  to_target_net_interval=10,
  device=DEVICE,
  dqn_type="DoubleDQN",
)

replay_buffer = dict(
  type="ReplayBuffer",
  capacity=10000,
  activate_size=500,
)

logger_cfg = dict(
  type="LoggerHook",
  log_interval=10,
  save_dir=work_dir,
  save_filename=exp_name,
)