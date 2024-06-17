import numpy as np

from labsurv.builders import ENVIRONMENTS

DANGER = 0
FREE = 1
START = 2
DEST = 3


@ENVIRONMENTS.register_module()
class CliffWalkModelFreeEnv:
    def __init__(self, world: list[int] | np.ndarray, reward):
        """
        Attributes:
          world (np.ndarray): the world grid of cliff walk env
        """
        if isinstance(world, list):
            world = np.array(world)
        if world.ndim != 2:
            raise ValueError("Cliff walk requires 2d world.")
        unique, counts = np.unique(world, return_counts=True)
        if len(set(unique).difference(set([DANGER, FREE, START, DEST]))) != 0:
            raise ValueError("`world` should only contain digits of 0 to 3.")
        if dict(zip(unique, counts))[START] != 1:
            raise ValueError("`world` should have one and only one START point.")
        if dict(zip(unique, counts))[DEST] != 1:
            raise ValueError("`world` should have one and only one DEST point.")

        self.world = world
        self.shape = world.shape

        start_pos = np.where(world == START)
        self.cur_state = (int(start_pos[0]), int(start_pos[1]))
        self.actions = [
            (-1, 0),  # NORTH
            (0, -1),  # WEST
            (1, 0),  # SOUTH
            (0, 1),  # EAST
        ]
        self.reward = reward

    def step(self, action):
        """
        The agent takes a single action by calling this method.
        """
        assert action in range(len(self.actions))

        new_state = (
            min(
                self.shape[0] - 1,
                max(0, self.cur_state[0] + self.actions[action][0]),
            ),
            min(
                self.shape[1] - 1,
                max(0, self.cur_state[1] + self.actions[action][1]),
            ),
        )
        self.cur_state = new_state
        reward = self.reward[int(self.world[new_state])]
        terminated = self.world[new_state] in [DANGER, DEST]

        return new_state, reward, terminated

    def reset(self):
        """
        Reset state to init.
        """
        start_pos = np.where(self.world == START)
        self.cur_state = (int(start_pos[0]), int(start_pos[1]))
        return self.cur_state
