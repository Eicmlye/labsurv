import numpy as np

from labsurv.builders import ENVIRONMENTS


DANGER = 0
FREE = 1
START = 2
DEST = 3


NORTH = 0
WEST = 1
SOUTH = 2
EAST = 3


@ENVIRONMENTS.register_module()
class CliffWalkEnv:
    def __init__(self, world: list[int] | np.ndarray, reward):
        if isinstance(world, list):
            world = np.array(world)

        if world.ndim != 2:
            raise ValueError("Cliff walk requires 2d world.")

        self.world = world
        self.shape = world.shape
        self.reward = reward
        self.transition = self.init_transition(world)
        self.actions = [NORTH, WEST, SOUTH, EAST]

    def init_transition(self, world: np.ndarray):
        """
        transition[state][action] == (prob, next_state, reward, done)
        """
        if len(set(np.unique(world)).difference(set([DANGER, FREE, START, DEST]))) != 0:
            raise ValueError("`world` should only contain digits of 0 to 3.")
        unique, count = np.unique(world, return_counts=True)
        grid_type_count = dict(zip(unique, count))
        if grid_type_count[START] != 1 or grid_type_count[DEST] != 1:
            raise ValueError("`world` requires 1 and only 1 START & DEST point.")

        transition = dict()
        for index in range(world.shape[0]):
            for jndex in range(world.shape[1]):
                cur_state = (index, jndex)
                neighbours = [
                    (index - 1, jndex),  # north
                    (index, jndex - 1),  # west
                    (index + 1, jndex),  # south
                    (index, jndex + 1),  # east
                ]

                probs = [1, 1, 1, 1]  # get_init_probs(world, neighbours)

                transition[cur_state] = {
                    NORTH: (
                        probs[0],
                        neighbours[0],
                        self.get_reward(world, cur_state, neighbours[0]),
                        check_done(world, cur_state, neighbours[0]),
                    ),
                    WEST: (
                        probs[1],
                        neighbours[1],
                        self.get_reward(world, cur_state, neighbours[1]),
                        check_done(world, cur_state, neighbours[1]),
                    ),
                    SOUTH: (
                        probs[2],
                        neighbours[2],
                        self.get_reward(world, cur_state, neighbours[2]),
                        check_done(world, cur_state, neighbours[2]),
                    ),
                    EAST: (
                        probs[3],
                        neighbours[3],
                        self.get_reward(world, cur_state, neighbours[3]),
                        check_done(world, cur_state, neighbours[3]),
                    ),
                }

        return transition

    def get_reward(self, world, pos, neighbour):
        """
        If walking from pos to neighbour is unavailable, return 0.
        Otherwise, return the reward from pos to neighbour.
        """
        if (
            world[pos] in [DANGER, DEST]
            or neighbour[0] not in range(world.shape[0])
            or neighbour[1] not in range(world.shape[1])
        ):
            return 0

        return self.reward[world[neighbour]]


def check_done(world, pos, neighbour):
    """
    If walking from pos to neighbour is unavailable, return None.
    Otherwise, check if neighbour is the end of the game.
    """
    if (
        world[pos] in [DANGER, DEST]
        or neighbour[0] not in range(world.shape[0])
        or neighbour[1] not in range(world.shape[1])
    ):
        return None

    return world[neighbour] in [DANGER, DEST]
