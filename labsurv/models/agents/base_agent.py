from typing import Any

from labsurv.builders import AGENTS, EXPLORERS
from torch import device


@AGENTS.register_module()
class BaseAgent:
    def __init__(
        self,
        device: device = None,
        gamma: float = 0.9,
        explorer_cfg=None,
    ):
        self.device = device
        self.gamma = gamma
        self.explorer = EXPLORERS.build(explorer_cfg)

    def take_action(self, observation: Any):
        # explore or exploit
        return self.explorer.act()

    def update(self, samples: dict):
        raise NotImplementedError()
