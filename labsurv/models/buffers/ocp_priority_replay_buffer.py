import os
import os.path as osp
import pickle
import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from labsurv.builders import REPLAY_BUFFERS
from labsurv.models.buffers import BaseReplayBuffer
from labsurv.utils.string import WARN
from numpy import ndarray as array


class SumTree:
    """
    Data structure for O(log n) priority experience replay.
    """

    def __init__(
        self, device: torch.cuda.device, capacity: int = 1, seed: int | None = None
    ):
        assert capacity > 0
        self.device = device

        self.capacity: int = capacity
        self.tree: List[float] = [0] * (2 * capacity)  # start index from 1
        self.data: List[Dict] = [None] * (capacity + 1)  # start index from 1
        self._write: int = 0  # the pred index of the last valid node

        self._random = random.Random(seed)

    def load(self, tree):
        if len(self) > 0:
            print(
                WARN(
                    "The replay buffer will be covered by loaded SumTree. "
                    "Current buffer will be lost. This operation is inreversible."
                )
            )

        if tree.capacity < self.capacity:
            self._write = len(tree)
        else:
            self._write = 0

        self.device = tree.device
        self._random = tree._random

        if tree.capacity <= self.capacity:
            self.tree[: len(tree) + self.capacity + 1] = tree.tree
            self.data[: len(tree) + self.capacity + 1] = tree.data
        else:
            self.tree = tree.tree[: len(self) + self.capacity + 1]
            self.data = tree.data[: len(self) + self.capacity + 1]

    def __len__(self):
        length = 0
        for index in range(1, self.capacity + 1):
            if self.data[index] is not None:
                length += 1

        return length

    def add(self, new_val, data):
        """
        Add a new node with value `new_val` and content `data`.
        """
        index = self._write + self.capacity
        self.data[self._write + 1] = data
        self._update(index, new_val)
        self._write += 1

        if self._write >= self.capacity:
            self._write = 0

    def _update(self, index: int, new_val: float):
        """
        Update the `index`-th node value to `new_val`.
        The entire tree will be updated automatically.
        """
        change = new_val - self.tree[index]
        self._propagate(index, change)
        self.tree[index] = new_val

    def _propagate(self, index: int, change: float):
        """
        Upper propagate the `change` that the `index`-th node has made.
        """
        parent = index // 2
        self.tree[parent] += change

        if parent != 1:
            self._propagate(parent, change)

    def get(self, val: float):
        index = self._retrieve(1, val)
        data_index = index - self.capacity + 1
        return (index, self.tree[index], self.data[data_index])

    def _retrieve(self, index: int, val: float) -> int:
        """
        Find the node index correspond to `val`,taking `index`-th node as the root.
        """
        lchild = 2 * index
        rchild = lchild + 1

        if lchild >= (len(self.tree) - 1):
            return index

        if val <= self.tree[lchild]:
            return self._retrieve(lchild, val)
        else:
            return self._retrieve(rchild, val - self.tree[lchild])

    def refresh(self, actor: torch.nn.Module, critic: torch.nn.Module, gamma: float):
        """
        Update the priority of all the nodes.
        """
        cache_td_error = []
        for data_index in range(1, self.capacity + 1):
            if self.data[data_index] is None:
                break

            with torch.no_grad():
                cur_observation = torch.tensor(
                    self.data[data_index]["cur_observation"],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                cur_action = torch.tensor(
                    self.data[data_index]["cur_action"],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                next_observation = torch.tensor(
                    self.data[data_index]["next_observation"],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                reward = torch.tensor(
                    self.data[data_index]["reward"],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                terminated = torch.tensor(
                    self.data[data_index]["terminated"],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                target_q = critic(next_observation, actor(next_observation))
                discounted_target_q = reward + gamma * target_q * (1 - terminated)
                td_error = discounted_target_q - critic(cur_observation, cur_action)
            cache_td_error.append(td_error.item())

        cache_prob = F.softmax(torch.tensor(cache_td_error, dtype=torch.float32), dim=0)

        for error_index in range(len(cache_td_error)):
            data_index = error_index + 1
            index = error_index + self.capacity

            self._update(index, cache_prob[error_index].item())

    def sample(self, batch_size: int) -> Dict[str, List[bool | float | array]]:
        """
        Get `batch_size` samples according to the priority distribution.
        """
        batch = []

        for batch_index in range(batch_size):
            batch.append(self.get(self._random.uniform(0, 1))[-1])

        return batch


@REPLAY_BUFFERS.register_module()
class OCPPriorityReplayBuffer(BaseReplayBuffer):
    """
    ## Description:

        A priority replay buffer for OCP problem.

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
        self.capacity = capacity

        if load_from is None:
            self._buffer = SumTree(device, self.capacity)
        else:
            self.load(load_from)

        self.activate_size = activate_size
        self.batch_size = batch_size

        self._random = random.Random(seed)

    def add(self, transition: Dict[str, bool | float | array]):
        self._buffer.add(0, transition)

    def update(self, actor: torch.nn.Module, critic: torch.nn.Module, gamma):
        """
        Update the priority of all the nodes.
        """
        self._buffer.refresh(actor, critic, gamma)

    def sample(self) -> Dict[str, List[bool | float | array]]:
        """
        Replay buffer is only available for sampling after the number of contents hit
        the threshold.
        """

        assert self.is_active(), "Sampling is not available for inactivate buffer."

        batch_transitions: List[Dict[str, bool | float | array]] = self._buffer.sample(
            self.batch_size
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
            loaded_buffer = pickle.load(f)

        if not isinstance(loaded_buffer, SumTree):
            raise RuntimeError("No SumTree structure detected.")
        if loaded_buffer.capacity == self.capacity:
            self._buffer = loaded_buffer
        else:
            self._buffer.load(loaded_buffer)

        print("Replay buffer loaded.")

    def save(self, save_path: str):
        if save_path.endswith(".pkl"):
            os.makedirs(osp.dirname(save_path), exist_ok=True)
        else:
            os.makedirs(save_path, exist_ok=True)
            save_path = osp.join(save_path, "replay_buffer.pkl")

        with open(save_path, "wb") as f:
            pickle.dump(self._buffer, f)
