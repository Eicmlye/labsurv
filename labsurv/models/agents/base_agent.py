from typing import Any

from labsurv.builders import AGENTS, EXPLORERS
from torch import device


@AGENTS.register_module()
class BaseAgent:
    def __init__(
        self,
        device: device = None,
        gamma: float = 0.9,
        explorer_cfg: dict = None,
    ):
        """
        ## Description:

            Base agent class for all agents.

        ## Arguments:

            device (torch.device): the device the agent will work on.

            gamma (float): the discount factor of the agent.

            explorer_cfg (dict): the configuration for the explorer.
        """
        self.device = device
        self.gamma = gamma
        if explorer_cfg is not None:
            self.explorer = EXPLORERS.build(explorer_cfg)

    def take_action(self, observation: Any):
        # explore or exploit
        return self.explorer.act()

    def update(self, samples: dict):
        raise NotImplementedError()
