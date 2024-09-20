from typing import List, Optional

import numpy as np
from labsurv.builders import EXPLORERS
from numpy import ndarray as array

from .base_explorer import BaseExplorer


@EXPLORERS.register_module()
class OCPEpsilonGreedyExplorer(BaseExplorer):
    def __init__(
        self,
        samples: int | float | List[int | float] = 2,
        samples_from: List[int | float | List[int | float]] = None,
        epsilon: float = 0.2,
        seed: Optional[int] = None,
    ):
        """
        ## Arguments:

            samples (int | float | List[int] | List[int | float | List[int]]):
            1. if is `int`, the explorer samples integers from [0, samples)
            2. if is `float`, the explorer samples real numbers uniformly from
                [0, samples)
            3. if is `List[int | float]`, the explorer returns a `len(samples)`-sized
                sample, of which every item is sampled as if the items of `samples` are
                the input respectively.
            4. if is None, use `samples_from`.

            samples_from (List[int | float | List[int | float]]):
            1. if `samples` is not None, this argument is ignored.
            2. if is `List[int | float]`, the explorer samples from the list `samples`
                itself.
        """
        super().__init__(seed)
        self.epsilon = epsilon

        self.samples = samples
        if samples is not None:
            if not isinstance(samples, (int, float, List)):
                raise ValueError(f"{type(samples)} is not allowed.")
        else:
            raise NotImplementedError()

    def decide(self):
        return self._random.uniform(0, 1) < self.epsilon

    def act(self):
        if self.samples is not None:
            if not isinstance(self.samples, List):
                print("This is not the usual case that an OCP problem uses.")
                return (
                    self._random.choice(self.samples)
                    if isinstance(self.samples, int)
                    else self._random.uniform(0, self.samples)
                )
            else:
                output = []
                for sample in self.samples:
                    if isinstance(sample, int):
                        output.append(self._random.choice(sample))
                    elif isinstance(sample, float):
                        output.append(self._random.uniform(0, sample))
                    else:
                        output.append(self._random.choice(sample))

                # the outer bracket unsqueezes the batch dimension
                return self._reformat_ocp_params(np.array([output], dtype=np.float32))
        else:
            raise NotImplementedError()

    def _reformat_ocp_params(self, params: array):
        """
        ## Returns:

            (action, pos_index, pan, tilt, cam_type) shaped and ranged array.
        """

        params[0, 2] *= 2 * np.pi
        params[0, 2] -= np.pi
        params[0, 3] *= np.pi
        params[0, 3] -= np.pi / 2

        return params
