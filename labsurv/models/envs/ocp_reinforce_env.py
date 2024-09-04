from typing import Dict, Optional, Tuple

import torch
from labsurv.builders import ENVIRONMENTS
from labsurv.models.envs import BaseSurveillanceEnv
from numpy import ndarray as array
from torch import Tensor


@ENVIRONMENTS.register_module()
class OCPREINFORCEEnv(BaseSurveillanceEnv):
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

        super().__init__(room_data_path, device, save_path)

    def step(
        self, observation: array, action: array, params: array, total_steps: int
    ) -> Tuple[Dict[str, float | array], float]:
        """
        ## Description:

            Run one timestep of the environment's dynamics.

        ## Arguments:

            observation (Tensor): [12, W, D, H], torch.float16, the info tensor.

            action (int): the action index.

            params (Tensor): [4], torch.float16,
            [pos_index_lambda, pan, tilt, cam_type_lambda]
        """
        ADD = 0
        DEL = 1
        ADJUST = 2
        STOP = 3
        self.action_count += 1

        observation: Tensor = torch.tensor(
            observation, dtype=self.FLOAT, device=self.device
        )
        action: int = torch.tensor(action, dtype=self.FLOAT, device=self.device).item()
        params: Tensor = torch.tensor(params, dtype=self.FLOAT, device=self.device)

        direction: array = (
            params[1:3]
            .clamp(
                torch.tensor(
                    [-torch.pi, -torch.pi / 2], dtype=self.FLOAT, device=self.device
                ),
                torch.tensor(
                    [torch.pi - 1e-5, torch.pi / 2],
                    dtype=self.FLOAT,
                    device=self.device,
                ),
            )
            .cpu()
            .numpy()
            .copy()
        )

        cam_type: int = (
            params[3]
            .clamp(
                torch.zeros([1], device=self.device),
                torch.tensor([len(self.info_room._CAM_TYPES) - 1], device=self.device),
            )
            .round()
            .type(self.INT)
            .item()
        )

        total_target_point_num: float = (
            self.info_room.must_monitor[:, :, :, 0].sum().item()
        )
        # pred_cam_count = self.info_room.cam_extrinsics[:, :, :, 0].sum().item()
        pred_coverage: float = (
            self.info_room.visible_points > 0
        ).sum().item() / total_target_point_num

        pred_cam_count = (
            self.info_room.cam_extrinsics[:, :, :, 0].sum().type(self.INT).item()
        )
        vis_mask = None
        if action == ADD:  # add cam
            # choose from uninstalled permitted pos
            candidates: array = (
                (
                    torch.logical_xor(
                        self.info_room.install_permitted,
                        self.info_room.cam_extrinsics[:, :, :, 0],
                    )
                    .nonzero()
                    .type(self.INT)
                )
                .cpu()
                .numpy()
                .copy()
            )
            pos_index: int = (
                (params[0] * (len(candidates) - 1))
                .clamp(
                    torch.zeros([1], device=self.device),
                    torch.tensor([len(candidates) - 1], device=self.device),
                )
                .round()
                .type(self.INT)
                .item()
            )
            pos: array = candidates[pos_index]

            print(
                f"Current coverage {pred_coverage:.2%} with {pred_cam_count:d} cameras. \n"
                f"Add camera at [{pos[0]}, {pos[1]}, {pos[2]}], "
                f"with pan={direction[0]:.4f}, tilt={direction[1]:.4f}, "
                f"type {cam_type} cam..."
            )
            vis_mask = self.info_room.add_cam(pos, direction, cam_type)
            print("\r\033[1A\033[K\033[1A\033[K", end="")
            self.history_cost[pos[0], pos[1], pos[2]] += 1
        elif action == DEL:  # del cam
            # choose from installed pos
            candidates: array = (
                (self.info_room.cam_extrinsics[:, :, :, 0].nonzero().type(self.INT))
                .cpu()
                .numpy()
                .copy()
            )
            pos_index: int = (
                (params[0] * (len(candidates) - 1))
                .clamp(
                    torch.zeros([1], device=self.device),
                    torch.tensor([len(candidates) - 1], device=self.device),
                )
                .round()
                .type(self.INT)
                .item()
            )
            pos: array = candidates[pos_index]

            print(
                f"Current coverage {pred_coverage:.2%} with {pred_cam_count:d} cameras. \n"
                f"Del camera at [{pos[0]}, {pos[1]}, {pos[2]}]..."
            )
            vis_mask = self.info_room.del_cam(pos)
            print("\r\033[1A\033[K\033[1A\033[K", end="")
            self.history_cost[pos[0], pos[1], pos[2]] += 1
        elif action == ADJUST:  # adjust cam
            # choose from installed pos
            candidates: array = (
                (self.info_room.cam_extrinsics[:, :, :, 0].nonzero().type(self.INT))
                .cpu()
                .numpy()
                .copy()
            )
            pos_index: int = (
                (params[0] * (len(candidates) - 1))
                .clamp(
                    torch.zeros([1], device=self.device),
                    torch.tensor([len(candidates) - 1], device=self.device),
                )
                .round()
                .type(self.INT)
                .item()
            )
            pos: array = candidates[pos_index]

            print(
                f"Current coverage {pred_coverage:.2%} with {pred_cam_count:d} cameras. \n"
                f"Mov camera at [{pos[0]}, {pos[1]}, {pos[2]}], "
                f"to pan={direction[0]:.4f}, tilt={direction[1]:.4f}, "
                f"type {cam_type} cam..."
            )
            vis_mask = self.info_room.adjust_cam(pos, direction, cam_type)
            print("\r\033[1A\033[K\033[1A\033[K", end="")
            self.history_cost[pos[0], pos[1], pos[2]] += 3
        elif action == STOP:  # stop
            print(
                f"Current coverage {pred_coverage:.2%} with {pred_cam_count:d} cameras. \n"
                "Stop modifying."
            )
            print("\r\033[1A\033[K\033[1A\033[K", end="")
        else:
            raise ValueError(f"Unknown action {action}.")

        cur_coverage: float = (
            self.info_room.visible_points > 0
        ).sum().item() / total_target_point_num

        if abs(cur_coverage - pred_coverage) <= 0.01 or action == ADJUST:
            self.lazy_count += 1
        else:
            self.lazy_count = 0

        cam_count = (
            self.info_room.cam_extrinsics[:, :, :, 0].sum().type(self.INT).item()
        )

        lambdas = [1, 1, 1]
        total_lambda = sum(lambdas)
        for index in range(len(lambdas)):
            lambdas[index] /= total_lambda

        reward = 0
        camera_threshold = 5
        if action != STOP:
            reward += (
                # camera reward
                vis_mask.sum() / total_target_point_num
                # coverage reward, the difference that vis_mask brings
                * (cur_coverage - pred_coverage)
            )
            
        if action == STOP or self.action_count == total_steps:
            reward += (
                # mission completion
                (cur_coverage - 1)
                # steps taken
                * self.action_count / total_steps
                # camera used
                * (1 if cam_count < camera_threshold else cam_count) / total_steps
            )

        transition = dict(
            next_observation=self.info_room.get_info(),
            reward=reward,
            terminated=(action == STOP),
        )

        return transition, cur_coverage, cam_count
