from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
from gym import spaces
from labsurv.builders import ENVIRONMENTS
from labsurv.models.envs import BaseEnv
from labsurv.physics import SurveillanceRoom
from labsurv.utils.surveillance import (
    DeleteUninstalledCameraError,
    InstallAtExistingCameraError,
)


@ENVIRONMENTS.register_module()
class BaseSurveillanceEnv(BaseEnv):
    def __init__(
        self,
        room_data_path: str,
    ):
        """
        ## Description:

            This environment is the basic surveillance room environment class.

        ## Action space:

            The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}`
            indicating the (un)installation of a camera.

            | Num | Action     |
            | --- | -----------|
            |  0  | add_cam    |
            |  1  | del_cam    |

        ## Observation space:

            The observation space is the attribute subspace of `SurveillanceRoom`.
            Specifically, the `cam_extrinsics`, `cam_types`, `visible_points` and
            `must_monitor` should be observable to compute the coverage of the current
            camera installation plan.

            To simplify the problem, the env uses an `SurvellanceRoom` object as a
            transition.

        ## Rewards:

            Rewards are simply represented by the coverage increment at current
            timestep. When a camera is installed or uninstalled, the coverage will
            increase or decrease accordingly, which gives the reward of the action at
            current state.

        ## Start state:

            No camera is installed at the very beginning.

        ## Episode End

            The episode truncates if the episode length is greater than 500.
        """

        super().__init__()

        self.init_visibility(room_data_path)

        self.action_space = spaces.Discrete(2)

    def step(
        self, observation: np.ndarray, action_with_params: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run one timestep of the environment's dynamics.

        `action` format:
        [action_id, pos_index_in_install_permitted, pan, tilt, cam_type]
        """
        action = action_with_params[0]
        params = (
            action_with_params[1:].astype(np.int16)
            if len(action_with_params) > 1
            else None
        )

        pred_coverage = len(self.info_room.visible_points) / len(
            self.info_room.must_monitor
        )

        if action == 0:  # add cam
            try:
                self.info_room.add_cam(
                    self.info_room.install_permitted[params[0]],
                    params[1:3],
                    params[3],
                )
            except InstallAtExistingCameraError:
                pass
        elif action == 1:  # del cam
            try:
                self.info_room.del_cam(self.info_room.install_permitted[params[0]])
            except DeleteUninstalledCameraError:
                pass
        else:
            raise ValueError(f"Unknown action {action}.")

        cur_coverage = len(self.info_room.visible_points) / len(
            self.info_room.must_monitor
        )
        reward = cur_coverage - pred_coverage

        transition = dict(
            next_observation=self.info_room.to_array(),
            reward=reward,
            terminated=False,
        )

        return transition

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """ """

        # do env init works
        super().reset(seed=seed)

        # return init observation according to observation distribution
        self.info_room = deepcopy(self.surv_room)

        return self.info_room.to_array()

    def init_visibility(self, room_data_path: str):
        """
        Load the surveillance room data. Caution that self.surv_room should never be
        modified. Any attempt to use self.surv_room must get a copy of the object.
        """
        self.surv_room = SurveillanceRoom(load_from=room_data_path)

    def render(self):
        pass
