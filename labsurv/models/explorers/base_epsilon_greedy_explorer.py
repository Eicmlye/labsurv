from typing import List, Optional

import numpy as np
from labsurv.builders import EXPLORERS
from numpy import ndarray as array

from .base_explorer import BaseExplorer


@EXPLORERS.register_module()
class BaseEpsilonGreedyExplorer(BaseExplorer):
    def __init__(
        self,
        samples: int | List[int | float] = 2,
        epsilon: float = 0.9,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        ## Arguments:

            samples (List[int]): the numbers to be sampled
        """
        super().__init__(seed)
        self.epsilon_max = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.samples = samples if isinstance(samples, int) else np.array(samples)

    def decide(self, observation: array):
        self._epsilon_update()
        return self._random.uniform(0, 1) < self.epsilon

    def act(self):
        return self._random.choice(self.samples)

    def _epsilon_update(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self):
        self.epsilon = self.epsilon_max
