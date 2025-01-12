import os
import os.path as osp
from copy import deepcopy
from typing import Optional

import torch
from labsurv.builders import ENVIRONMENTS
from labsurv.models.envs import BaseEnv
from labsurv.physics import SurveillanceRoom
from labsurv.utils.string import to_filename
from numpy import ndarray as array


@ENVIRONMENTS.register_module()
class BaseSurveillanceEnv(BaseEnv):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        room_data_path: str,
        device: Optional[str],
        save_path: str,
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
            The room class will merge all the tensor attributes as a single tensor of
            shape [12, W, D, H] and send it out as the info flow.

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

        self.device = torch.device(device)
        self.save_path = save_path
        self.action_count = 0
        self.lazy_count = 0
        self.init_visibility(room_data_path)

        self.history_cost = torch.zeros_like(
            self._surv_room.occupancy, dtype=self.INT, device=self.device
        )

    def step(self, observation: array, action: int, params: array, total_steps: int):
        """
        ## Description:

            Run one timestep of the environment's dynamics.

        ## Arguments:

            observation (Tensor): [12, W, D, H], torch.float16, the info tensor.

            action (int): the action index.

            params (Tensor): [4], torch.float16,
            [pos_index_lambda, pan, tilt, cam_type_lambda]
        """

        raise NotImplementedError()

    def reset(self, seed: Optional[int] = None) -> array:
        """
        ## Description:

            Reset the environment to the initial state and return the env info.

        ## Returns:

            A `Tensor` merged by all the tensor attributes of the room.
        """

        # do env init works
        super().reset(seed=seed)

        # return init observation according to observation distribution
        del self.info_room
        self.info_room = deepcopy(self._surv_room)

        self.action_count = 0
        self.lazy_count = 0
        self.history_cost = torch.zeros_like(
            self._surv_room.occupancy, dtype=self.INT, device=self.device
        )

        return self.info_room.get_info()

    def init_visibility(self, room_data_path: str):
        """
        Load the surveillance room data. Caution that self._surv_room should never be
        modified. Any attempt to use self._surv_room must get a copy of the object.
        """
        self._surv_room = SurveillanceRoom(device=self.device, load_from=room_data_path)
        self.info_room = None

    def render(self, observation: array, step: int):
        os.makedirs(self.save_path, exist_ok=True)
        cur_step_save_path = osp.join(self.save_path, f"step_{step + 1}")

        self.info_room.save(to_filename(cur_step_save_path, "pkl", "SurveillanceRoom"))
        self.info_room.visualize(
            to_filename(cur_step_save_path, "ply", "SurveillanceRoom")
        )
        self.info_room.visualize(
            to_filename(cur_step_save_path, "ply", "SurveillanceRoom"), "camera"
        )
