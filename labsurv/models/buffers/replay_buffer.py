import collections
import numpy as np
import random


class ReplayBuffer:
  def __init__(self, capacity: int):
    self._buffer = collections.deque(maxlen=capacity) 

  def add(
    self,
    state,
    action: int,
    reward: float,
    next_state,
    terminated: bool,
  ): 
    self._buffer.append((state, action, reward, next_state, terminated)) 

  def sample(self, batch_size): 
    transitions = random.sample(self._buffer, batch_size)
    state, action, reward, next_state, terminated = zip(*transitions)
    return np.array(state), action, reward, np.array(next_state), terminated 

  def size(self): 
    return len(self._buffer)