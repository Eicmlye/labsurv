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
        pan_section_num: int = 360,
        tilt_section_num: int = 180,
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
        self._reward = 0
        self.pan_section_num = pan_section_num
        self.tilt_section_num = tilt_section_num

        self.samples = samples
        if samples is not None:
            if not isinstance(samples, (int, float, List)):
                raise ValueError(f"{type(samples)} is not allowed.")
        else:
            raise NotImplementedError()

    def decide(self, observation: array) -> bool:
        return self._random.uniform(0, 1) < self.epsilon

    def act(self, observation: array) -> array:
        """
        ## Returns:

            action_with_params (np.ndarray): B * 5, pos_index, not pos_coord
        """
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
                # action
                add_act = None
                if self.samples[0] == 1:
                    add_act = 0
                    output.append(0)
                else:
                    if (observation[7] != 0).sum() == 0:
                        add_act = 0
                    elif (observation[1] != 0).sum() == (observation[7] != 0).sum():
                        add_act = 1
                    else:
                        add_act = self._random.choice(2)

                    if add_act == 1:
                        output.append(self._random.choice(2) + 1)
                    else:
                        output.append(0)

                # pos_index, pan_index, tilt_index, cam_type
                for index, sample in enumerate(self.samples[1:]):
                    if index == 0:
                        # only choose pos in permitted area
                        permitted_zone = None
                        if add_act == 1:
                            permitted_zone = observation[7].flatten().nonzero()[0]
                        else:
                            permitted_zone = (
                                np.logical_xor(observation[1], observation[7])
                                .flatten()
                                .nonzero()[0]
                            )
                        output.append(
                            permitted_zone[self._random.choice(len(permitted_zone))]
                        )
                    elif isinstance(sample, int):
                        output.append(self._random.choice(sample))
                    elif isinstance(sample, float):
                        output.append(self._random.uniform(0, sample))
                    else:
                        output.append(self._random.choice(sample))

                # the outer bracket unsqueezes the batch dimension
                return self._reformat_ocp_params(np.array([output], dtype=np.float32))
        else:
            raise NotImplementedError()

    def _reformat_ocp_params(self, params: array) -> array:
        """
        ## Returns:

            (action, pos_index, pan, tilt, cam_type) shaped and ranged array.
        """

        params[0, 2] = (
            (params[0, 2] - self.pan_section_num / 2) * 2 * np.pi / self.pan_section_num
        )
        params[0, 3] = (
            (params[0, 3] - self.tilt_section_num / 2) * np.pi / self.tilt_section_num
        )

        return params
