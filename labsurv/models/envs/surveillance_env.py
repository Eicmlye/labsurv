from typing import Any, Dict, Optional

from labsurv.builders import ENVIRONMENTS
from labsurv.models.envs import BaseEnv


@ENVIRONMENTS.register_module()
class SurveillanceEnv(BaseEnv):
    def __init__(self):
        """
        
        """
        # should clarify the init observation distribution in convenience of `reset()`
        raise NotImplementedError()

    def step(self, observation, action) -> Dict[str, Any]:
        """
        
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

    def reset(self, seed: Optional[int] = None):
        """
        
        """

        # do env init works
        super().reset(seed=seed)

        # return init observation according to observation distribution
        init_observation = None

        return init_observation
