from labsurv.builders import LOSS
from torch import Tensor, mean
import torch.nn.functional as F


@LOSS.register_module()
class TDLoss:
  def __init__(self):
    self.loss = F.mse_loss

  def get_loss(self, total_reward: Tensor, q_target: Tensor):
    return mean(self.loss(total_reward, q_target))