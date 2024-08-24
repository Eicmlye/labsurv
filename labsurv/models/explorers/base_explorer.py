from typing import Optional

from labsurv.builders import EXPLORERS
from labsurv.utils.random import np_random


@EXPLORERS.register_module()
class BaseExplorer:
    def __init__(self, seed: Optional[int] = None):
        self._random, _ = np_random(seed)

    def decide(self):
        """
        Decide either to explore or to exploit.
        """

        raise NotImplementedError

    def act(self):
        """
        Actions when exploring.
        """

        raise NotImplementedError
