import collections
import random

from labsurv.builders import REPLAY_BUFFERS


@REPLAY_BUFFERS.register_module()
class ReplayBuffer(object):
    def __init__(self, capacity: int, activate_size: int):
        if activate_size > capacity:
            raise ValueError(
                "Replay buffer is never activated when activate_size "
                f"{activate_size} is greater than capacity {capacity}."
            )
        self._buffer = collections.deque(maxlen=capacity)
        self.activate_size = activate_size

    def add(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        terminated: bool,
        truncated: bool,
    ):
        self._buffer.append((state, action, reward, next_state, terminated, truncated))

    def sample(self, batch_size):
        assert self.is_active(), "Sampling is not available for inactivate buffer."
        assert batch_size < len(self), (
            f"batchsize {batch_size} should be no greater than "
            f"current buffer size {len(self)}."
        )

        transitions = random.sample(self._buffer, batch_size)
        (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_terminated,
            batch_truncated,
        ) = zip(*transitions)

        transitions = dict(
            states=batch_states,
            actions=batch_actions,
            rewards=batch_rewards,
            next_states=batch_next_states,
            terminated=batch_terminated,
            truncated=batch_truncated,
        )

        return transitions

    def __len__(self):
        return len(self._buffer)

    def is_active(self):
        return len(self) >= self.activate_size
