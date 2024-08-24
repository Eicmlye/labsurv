from typing import List, Optional

from labsurv.builders import EXPLORERS

from .base_explorer import BaseExplorer


@EXPLORERS.register_module()
class BaseEpsilonGreedyExplorer(BaseExplorer):
    def __init__(
        self,
        samples: List[int],
        epsilon: float = 0.2,
        seed: Optional[int] = None,
    ):
        """
        ## Arguments:

            samples (List[int]): the numbers to be sampled
        """
        super().__init__(seed)
        self.epsilon = epsilon

        self.samples = samples

    def decide(self):
        return self._random.uniform(0, 1) < self.epsilon

    def act(self):
        return self._random.choice(self.samples)
