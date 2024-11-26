from typing import Optional

from labsurv.builders import EXPLORERS
from labsurv.utils.random import np_random
from numpy import ndarray as array


@EXPLORERS.register_module()
class BaseExplorer:
    def __init__(self, seed: Optional[int] = None):
        self._random, _ = np_random(seed)
        self._reward = 1

    def decide(self, observation: array):
        """
        Decide either to explore or to exploit.
        """

        raise NotImplementedError()

    def act(self):
        """
        Actions when exploring.
        """

        raise NotImplementedError()

    @property
    def reward(self) -> float:
        """
        Intrinsic reward factor if needed.
        """

        return self._reward
