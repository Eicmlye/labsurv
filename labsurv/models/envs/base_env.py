from typing import Any, Dict

from labsurv.builders import ENVIRONMENTS
from labsurv.utils.random import np_random


@ENVIRONMENTS.register_module()
class BaseEnv:
    def __init__(self, seed: int | None = None):
        """
        ## Description:

            In order to be compatible with the runner class, the outputs of some methods
            must be dicts with a minimal set of keys.

        ## Attributes:

            actions: the representation of action space

            observations: the representation of observation space
        """
        # should clarify the init observation distribution in convenience of `reset()`

        self.seed = seed
        self._np_random, _ = np_random(self.seed)

    def step(self, observation, action) -> Dict[str, Any]:
        """
        ## Description:

            Run one timestep of the environment's dynamics.

            This requires current observation transfer from agent, so that environment do
            not have to store `current observations`, which further decouples agents and
            environments.

            By doing so, the environment class allows monitoring dynamic changes during
            the learning process.

        ## Returns:

            a dict with minimal set of keys as follows:
                next_observation,
                reward,
                terminated,
                truncated,
        """
        assert action in self.action_space, "Unknown action."
        assert observation in self.observation_space, "Unknown input observation."

        transition = dict(
            next_observation=None,
            reward=None,
            terminated=None,
            truncated=None,
        )

        assert (
            transition["next_observation"] is None
            or transition["next_observation"] in self.observation_space
        ), "Unknown output observation."

        return transition

    def reset(self, seed: int | None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """

        # do env init works
        self._np_random, _ = np_random(self.seed if seed == -1 else seed)

        # return init observation according to observation distribution
        init_observation = None

        return init_observation

    def render(self):
        """
        Visualization related operations.
        """
        raise NotImplementedError()
