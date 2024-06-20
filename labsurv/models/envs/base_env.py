from typing import Any, Dict

import gym
from labsurv.builders import ENVIRONMENTS


@ENVIRONMENTS.register_module()
class BaseEnv(gym.Env):
    def __init__(self, actions: Any, observations: Any, reward_range: Any = None):
        """
        In order to be compatible with the runner class, the outputs of some methods
        must be dicts with a minimal set of keys.

        Attributes:
            actions: the representation of action space
            observations: the representation of observation space
        """

        self.action_space = actions
        # should clarify the init observation distribution in convenience of `reset()`
        self.observation_space = observations
        if reward_range is not None:
            self.reward_range = reward_range

    def step(self, observation, action) -> Dict[str, Any]:
        """
        Run one timestep of the environment's dynamics.

        This requires current observation transfer from agent, so that environment do
        not have to store `current observations`, which further decouples agents and
        environments.

        By doing so, the environment class allows monitoring dynamic changes during
        the learning process.

        Returns a dict with minimal set of keys as follows:
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        """
        assert action in self.action_space, "Unknown action."
        assert observation in self.observation_space, "Unknown input observation."

        transitions = dict(
            next_observation=None,
            reward=None,
            terminated=None,
            truncated=None,
            info=None,
        )

        assert (
            transitions["next_observation"] is None
            or transitions["next_observation"] in self.observation_space
        ), "Unknown output observation."

        return transitions

    def reset(self):
        """
        Resets the environment to an initial state and returns the initial observation.
        """

        # do env init works

        # return init observation according to observation distribution
        init_observation = None

        return init_observation
