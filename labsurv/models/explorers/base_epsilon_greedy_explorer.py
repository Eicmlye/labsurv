from labsurv.builders import EXPLORERS
from labsurv.models.explorers import BaseExplorer


@EXPLORERS.register_module()
class BaseEpsilonGreedyExplorer(BaseExplorer):
    def __init__(self, epsilon: float = 0.2, seed: int | None = None):
        self.epsilon = epsilon

        super().__init__(seed)

    def decide(self):
        return self._random.uniform(0, 1) < self.epsilon
