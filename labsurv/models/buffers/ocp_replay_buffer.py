import collections
import os
import os.path as osp
import pickle
import random
from typing import Dict, List

import torch
from labsurv.builders import REPLAY_BUFFERS
from numpy import ndarray as array


@REPLAY_BUFFERS.register_module()
class OCPReplayBuffer:
    """
    ## Description:

        A replay buffer for OCP problem.

    ## Items:

        sample (Dict[str, float | array | Tuple[int, array]]): the (s, a, r, s') tuple
        of a single step.

    ## Samples:

        The `sample()` method returns certain numbers of items sampled from the replay
        buffer uniformly. They are reformatted to (Dict[str, Tensor]), where
        `sample()[key][index]` is `key` value of the `index`-th sample.
    """

    def __init__(
        self,
        device: torch.cuda.device,
        batch_size: int,
        capacity: int,
        activate_size: int,
        seed: int | None = None,
        load_from: str | None = None,
    ):
        if activate_size > capacity:
            raise ValueError(
                "Replay buffer will never be activated when activate_size "
                f"{activate_size} is greater than capacity {capacity}."
            )
        if batch_size > capacity:
            raise ValueError(
                f"Batch size {batch_size} should be no greater than "
                f"buffer capacity {capacity}."
            )

        self.device = device

        if load_from is None:
            self._buffer = collections.deque(maxlen=capacity)
        else:
            self.load(load_from)

        self.activate_size = activate_size
        self.batch_size = batch_size

        self._random = random.Random(seed)

    def add(self, transition: Dict[str, bool | float | array]):
        self._buffer.append(transition)

    def sample(self) -> Dict[str, List[bool | float | array]]:
        """
        Replay buffer is only available for sampling after the number of contents hit
        the threshold.
        """

        assert self.is_active(), "Sampling is not available for inactivate buffer."

        batch_transitions: List[Dict[str, bool | float | array]] = self._random.sample(
            self._buffer, self.batch_size
        )

        samples: Dict[str, List[bool | float | array]] = {
            key: [] for key in batch_transitions[0].keys()
        }
        for transition in batch_transitions:
            for key, val in transition.items():
                samples[key].append(val)

        return samples

    def __len__(self):
        return len(self._buffer)

    def is_active(self) -> bool:
        return len(self) >= self.activate_size

    def load(self, load_from: str):
        with open(load_from, "rb") as f:
            self._buffer = pickle.load(f)
        print("Replay buffer loaded.")

    def save(self, save_path: str):
        if save_path.endswith(".pkl"):
            os.makedirs(osp.dirname(save_path), exist_ok=True)
        else:
            os.makedirs(save_path, exist_ok=True)
            save_path = osp.join(save_path, "replay_buffer.pkl")

        with open(save_path, "wb") as f:
            pickle.dump(self._buffer, f)