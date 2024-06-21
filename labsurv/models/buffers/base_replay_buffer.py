import collections
import random

from labsurv.builders import REPLAY_BUFFERS


@REPLAY_BUFFERS.register_module()
class BaseReplayBuffer:
    def __init__(
        self,
        batch_size: int,
        capacity: int,
        activate_size: int,
        seed: int | None = None,
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

        self._buffer = collections.deque(maxlen=capacity)
        self.activate_size = activate_size
        self.batch_size = batch_size

        self._random = random.Random(seed)

    def add(self, transition: dict):
        self._buffer.append(transition)

    def sample(self):
        """
        Replay buffer is only available for sampling after the number of contents hit
        the threshold.
        """

        assert self.is_active(), "Sampling is not available for inactivate buffer."

        batch_transitions = self._random.sample(self._buffer, self.batch_size)

        samples = {key: [] for key in batch_transitions[0].keys()}
        for transition in batch_transitions:
            for key, val in transition.items():
                samples[key].append(val)

        return samples

    def __len__(self):
        return len(self._buffer)

    def is_active(self):
        return len(self) >= self.activate_size
