from typing import Dict, Optional

# import numpy as np
import torch
import torch.nn.functional as F
from labsurv.builders import EXPLORERS
from labsurv.builders.builder import STRATEGIES
from numpy import ndarray as array
from torch.nn import Module

from .base_explorer import BaseExplorer


@EXPLORERS.register_module()
class OCPRNDExplorer(BaseExplorer):
    def __init__(
        self,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        lr: float = 0.1,
        net_cfg: Dict = None,
    ):
        """
        ## Arguments:


        """
        super().__init__(seed)

        self.device = torch.device(device)
        self.lr: float = lr

        self.teacher: Module = STRATEGIES.build(net_cfg).to(self.device)
        self.student: Module = STRATEGIES.build(net_cfg).to(self.device)

        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)

        # freeze teacher params
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

        self._reward: float = 1.0

    def decide(self, observation: array) -> bool:
        # RND uses policy net itself as action generator.
        # The intrinsic reward impacts the probability of action.

        self._reward = self._check_curiosity(observation)

        return False

    def _check_curiosity(self, observation: array) -> float:
        obs = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        rnd_loss = F.mse_loss(self.student(obs), self.teacher(obs))

        self.optimizer.zero_grad()
        rnd_loss.backward()
        self.optimizer.step()

        return torch.sigmoid(rnd_loss).item()

    @property
    def reward(self) -> float:
        """
        Returns:

        reward (float): an intrinsic reward ranged [0, 1]
        """
        return self._reward
