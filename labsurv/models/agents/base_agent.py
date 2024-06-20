from typing import Any

from labsurv.builders import AGENTS
from torch import device


@AGENTS.register_module()
class BaseAgent:
    def __init__(
        self,
        device: device = None,
        gamma: float = 0.9,
    ):
        self.device = device
        self.gamma = gamma

    def take_action(self, observation: Any):
        # explore or exploit
        raise NotImplementedError()

    def update(self, **transition):
        missing_keys = set(
            [
                "cur_obseravtion",
                "cur_action",
                "rewards",
                "next_obseravtion",
                "terminated",
                "truncated",
            ]
        ).difference(set(transition.keys()))
        assert len(missing_keys) == 0, f"Missing keys {missing_keys}."
