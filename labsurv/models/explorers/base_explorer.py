from labsurv.builders import EXPLORERS
from labsurv.utils.random import np_random


@EXPLORERS.register_module()
class BaseExplorer:
    def __init__(self, seed: int | None = None):
        self._random, _ = np_random(seed)

    def act(self):
        # should decide either to explore or to exploit
        raise NotImplementedError

    def explore(self):
        raise NotImplementedError

    def exploit(self):
        raise NotImplementedError
