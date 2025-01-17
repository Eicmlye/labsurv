from typing import List, Optional

import numpy as np
from labsurv.builders import EXPLORERS
from labsurv.utils.surveillance import direction_index2pan_tilt, pos_index2coord
from numpy import ndarray as array

from .base_explorer import BaseExplorer


@EXPLORERS.register_module()
class OCPEpsilonGreedyExplorer(BaseExplorer):
    def __init__(
        self,
        action_num: int,
        room_shape: List[int],
        cam_types: int,
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
        self.action_num = action_num
        self.room_shape = room_shape
        self.pan_section_num = pan_section_num
        self.tilt_section_num = tilt_section_num
        self.cam_types = cam_types

    def decide(self, observation: array) -> bool:
        return self._random.uniform(0, 1) < self.epsilon

    def act(self, observation: array) -> array:
        """
        ## Returns:

            action_with_params (np.ndarray): [9].
        """

        # action
        output = [self._random.choice(self.action_num)]

        # pos_index
        permitted_zone = (
            np.logical_xor(observation[1], observation[7]).flatten().nonzero()[0]
        )
        pos_index = permitted_zone[self._random.choice(len(permitted_zone))]
        output += pos_index2coord(self.room_shape, pos_index).tolist()

        direction_index = self._random.choice(
            self.pan_section_num * (self.tilt_section_num - 1) + 1
        )
        output += direction_index2pan_tilt(
            self.pan_section_num,
            self.tilt_section_num,
            direction_index,
        ).tolist()

        # cam_type
        output.append(self._random.choice(self.cam_types))

        output += [pos_index, direction_index]

        return np.array(output, dtype=np.float32)  # [9]
