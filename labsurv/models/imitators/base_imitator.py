import math
from typing import Optional

import torch
from labsurv.builders import IMITATORS
from torch import pi as PI


@IMITATORS.register_module()
class BaseImitator:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device)

    def update_scheduler(self, cur_episode: int, total_episode: int, mode: str = "cos"):
        """
        ## Description:

            Cosine one-cycle scheduler.
        """
        cur_episode = min(cur_episode, total_episode - 1)

        if mode == "cos":
            if isinstance(self.lr, list):
                for index in range(len(self.lr)):
                    time_index = PI / 4 + PI * 5 / 4 * cur_episode / (total_episode - 1)
                    min_lr = 1e-2 * self.max_lr[index]
                    amplification = (self.max_lr[index] - min_lr) / 2

                    self.lr[index] = amplification * (math.sin(time_index) + 1) + min_lr
            else:
                time_index = PI / 4 + PI * 5 / 4 * cur_episode / (total_episode - 1)
                min_lr = 1e-2 * self.max_lr
                amplification = (self.max_lr - min_lr) / 2

                self.lr = amplification * (math.sin(time_index) + 1) + min_lr
        else:
            raise NotImplementedError()
