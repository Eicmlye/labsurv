import os
import os.path as osp
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
from labsurv.builders import ENVIRONMENTS
from labsurv.models.envs import BaseEnv
from labsurv.physics import SurveillanceRoom
from labsurv.utils.string import to_filename
from torch import Tensor


@ENVIRONMENTS.register_module()
class BaseSurveillanceEnv(BaseEnv):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        room_data_path: str,
        device: Optional[torch.cuda.device],
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

        self.device = device
        self.save_path = save_path
        self.init_visibility(room_data_path)

        self.history_cost = torch.zeros_like(
            self.surv_room.occupancy, dtype=self.INT, device=self.device
        )

    def step(
        self, observation: Tensor, action: int, params: Tensor
    ) -> Tuple[Dict[str, Any], float]:
        """
        ## Description:

            Run one timestep of the environment's dynamics.

        ## Arguments:

            observation (Tensor): [12, W, D, H], torch.float16, the info tensor.

            action (int): the action index.

            params (Tensor): [4], torch.float16,
            [pos_index_lambda, pan, tilt, cam_type_lambda]
        """

        direction = params[1:3].clamp(
            torch.tensor(
                [-torch.pi, -torch.pi / 2], dtype=self.FLOAT, device=self.device
            ),
            torch.tensor(
                [torch.pi - 1e-5, torch.pi / 2], dtype=self.FLOAT, device=self.device
            ),
        )

        cam_type = (
            params[3]
            .clamp(
                torch.zeros([1], device=self.device),
                torch.tensor([len(self.info_room._CAM_TYPES) - 1], device=self.device),
            )
            .round()
            .type(self.INT)
            .item()
        )

        total_target_point_num = self.info_room.must_monitor[:, :, :, 0].sum().item()
        # pred_cam_count = self.info_room.cam_extrinsics[:, :, :, 0].sum().item()
        pred_coverage = (
            self.info_room.visible_points > 0
        ).sum().item() / total_target_point_num

        if action == 0:  # add cam
            # choose from uninstalled permitted pos
            candidates = (
                torch.logical_xor(
                    self.info_room.install_permitted,
                    self.info_room.cam_extrinsics[:, :, :, 0],
                )
                .nonzero()
                .type(self.INT)
            )
            pos_index = (
                (params[0] * (len(candidates) - 1))
                .clamp(
                    torch.zeros([1], device=self.device),
                    torch.tensor([len(candidates) - 1], device=self.device),
                )
                .round()
                .type(self.INT)
                .item()
            )
            pos = candidates[pos_index].type(self.INT)

            print(
                f"Current coverage {pred_coverage:.2%}. \nAdd camera at "
                f"[{pos[0].item()}, {pos[1].item()}, {pos[2].item()}], "
                f"with pan={direction[0].item():.4f}, tilt={direction[1].item():.4f}, "
                f"type {cam_type} cam..."
            )
            self.info_room.add_cam(pos, direction, cam_type)
            print("\r\033[1A\033[K\033[1A\033[K", end="")
            self.history_cost[pos[0], pos[1], pos[2]] += 1
        elif action == 1:  # del cam
            # choose from installed pos
            candidates = (
                self.info_room.cam_extrinsics[:, :, :, 0].nonzero().type(self.INT)
            )
            pos_index = (
                (params[0] * (len(candidates) - 1))
                .clamp(
                    torch.zeros([1], device=self.device),
                    torch.tensor([len(candidates) - 1], device=self.device),
                )
                .round()
                .type(self.INT)
                .item()
            )
            pos = candidates[pos_index].type(self.INT)

            print(
                f"Current coverage {pred_coverage:.2%}. \nDel camera at "
                f"[{pos[0].item()}, {pos[1].item()}, {pos[2].item()}]..."
            )
            self.info_room.del_cam(pos)
            print("\r\033[1A\033[K\033[1A\033[K", end="")
            self.history_cost[pos[0], pos[1], pos[2]] += 1
        elif action == 2:  # adjust cam
            # choose from installed pos
            candidates = (
                self.info_room.cam_extrinsics[:, :, :, 0].nonzero().type(self.INT)
            )
            pos_index = (
                (params[0] * (len(candidates) - 1))
                .clamp(
                    torch.zeros([1], device=self.device),
                    torch.tensor([len(candidates) - 1], device=self.device),
                )
                .round()
                .type(self.INT)
                .item()
            )
            pos = candidates[pos_index].type(self.INT)

            print(
                f"Current coverage {pred_coverage:.2%}. \nMov camera from "
                f"[{pos[0].item()}, {pos[1].item()}, {pos[2].item()}], "
                f"to pan={direction[0].item():.4f}, tilt={direction[1].item():.4f}, "
                f"type {cam_type} cam..."
            )
            self.info_room.adjust_cam(pos, direction, cam_type)
            print("\r\033[1A\033[K\033[1A\033[K", end="")
            self.history_cost[pos[0], pos[1], pos[2]] += 3
        else:
            raise ValueError(f"Unknown action {action}.")

        # cur_cam_count = self.info_room.cam_extrinsics[:, :, :, 0].sum().item()
        cur_coverage = (
            self.info_room.visible_points > 0
        ).sum().item() / total_target_point_num

        lambdas = [0.5, 0.25, 0.5]
        reward = (
            # > part 1: 100% at most
            lambdas[0]
            # rewards positive coverage changes
            * (cur_coverage - pred_coverage)
            # punish frequently adjustment on the same position
            / self.history_cost[pos[0], pos[1], pos[2]].item()
            # > part 2: -100% at most
            # basically the normalization terms
            - lambdas[1] * (
                torch.abs(params[0] - 0.5) * 2
                + torch.abs(params[1]) / torch.pi
                + torch.abs(params[2]) / torch.pi * 2
                + torch.abs(params[3] - 0.5) * 2
            ).item() / 4
            # > part 3: 100% at most
            # the total coverage
            + lambdas[2] * cur_coverage
        )

        transition = dict(
            next_observation=self.info_room.get_info(),
            reward=reward,
            terminated=False,
        )

        return transition, cur_coverage

    def reset(self, seed: Optional[int] = None) -> Tensor:
        """
        ## Description:

            Reset the environment to the initial state and return the env info.

        ## Returns:

            A `Tensor` merged by all the tensor attributes of the room.
        """

        # do env init works
        super().reset(seed=seed)

        # return init observation according to observation distribution
        self.info_room = deepcopy(self.surv_room)

        return self.info_room.get_info()

    def init_visibility(self, room_data_path: str):
        """
        Load the surveillance room data. Caution that self.surv_room should never be
        modified. Any attempt to use self.surv_room must get a copy of the object.
        """
        self.surv_room = SurveillanceRoom(device=self.device, load_from=room_data_path)

    def render(self, observation: Tensor, step: int):
        os.makedirs(self.save_path, exist_ok=True)
        cur_step_save_path = osp.join(self.save_path, f"step_{step}")

        self.info_room.save(to_filename(cur_step_save_path, "pkl", "SurveillanceRoom"))
        self.info_room.visualize(
            to_filename(cur_step_save_path, "ply", "SurveillanceRoom")
        )
        self.info_room.visualize(
            to_filename(cur_step_save_path, "ply", "SurveillanceRoom"), "camera"
        )
