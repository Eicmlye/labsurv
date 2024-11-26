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
        epsilon: float = 0.2,
        seed: Optional[int] = None,
    ):
        """
        ## Arguments:

            samples (List[int]): the numbers to be sampled
        """
        super().__init__(seed)
        self.epsilon = epsilon

        self.samples = samples if isinstance(samples, int) else np.array(samples)

    def decide(self, observation: array):
        return self._random.uniform(0, 1) < self.epsilon

    def act(self):
        return self._random.choice(self.samples)
