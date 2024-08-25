from copy import deepcopy
from typing import Any, Dict, Optional

from gym import spaces

import numpy as np
from labsurv.builders import ENVIRONMENTS
from labsurv.models.envs import BaseEnv
from labsurv.physics import SurveillanceRoom


@ENVIRONMENTS.register_module()
class BaseSurveillanceEnv(BaseEnv):
    def __init__(
        self,
        room_data_path: str,
    ):
        """ """

        self.init_visibility(room_data_path)

        # should clarify the init observation distribution in convenience of `reset()`
        raise NotImplementedError()

    def step(self, observation, action) -> Dict[str, Any]:
        """ """
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

    def reset(self, seed: Optional[int] = None) -> SurveillanceRoom:
        """ """

        # do env init works
        super().reset(seed=seed)

        # return init observation according to observation distribution
        surv_room = deepcopy(self.surv_room)

        return surv_room

    def init_visibility(self, room_data_path: str):
        """
        Load the surveillance room data. Caution that self.surv_room should never be
        modified. Any attempt to use self.surv_room must get a copy of the object.
        """
        self.surv_room = SurveillanceRoom(load_from=room_data_path)

    def render(self):
        pass
