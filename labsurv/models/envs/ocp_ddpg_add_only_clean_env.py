from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from labsurv.builders import ENVIRONMENTS
from labsurv.models.envs import BaseSurveillanceEnv
from labsurv.physics.surveillance_room import if_need_obstacle_check
from numpy import ndarray as array


@ENVIRONMENTS.register_module()
class OCPDDPGAddOnlyCleanEnv(BaseSurveillanceEnv):
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

            This environment is the DDPG surveillance room environment class.
            Only ADD action is allowed.

        ## Action space:

            The action is a `ndarray` with shape `(1,)` which only takes value `{0}`
            indicating the installation of a camera.

            | Num | Action     | Parameters               |
            | --- | ---------- | ------------------------ |
            |  0  | add_cam    | pos, direction, cam_type |

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
        self,
        observation: array,
        action_with_params: array,
        total_steps: int,
        section_nums: List[int],
    ) -> Tuple[Dict[str, bool | float | array], float, float]:
        """
        ## Description:

            Run one timestep of the environment's dynamics.

        ## Arguments:

            observation (Tensor): [12, W, D, H], torch.float16, the info tensor.

            action (int): the action index.

            params (Tensor): [4], torch.float16,
            [pos_index_lambda, pan, tilt, cam_type_lambda]
        """
        ILLEGAL = -1
        ADD = 0
        STOP = 3
        self.action_count += 1

        action = action_with_params[0].astype(np.int64)
        pos = action_with_params[1:4].astype(np.int64)
        direction = action_with_params[4:6]
        cam_type = action_with_params[6].astype(np.int64)

        total_target_point_num: float = (
            self.info_room.must_monitor[:, :, :, 0].sum().item()
        )
        pred_coverage: float = (
            self.info_room.visible_points > 0
        ).sum().item() / total_target_point_num

        # pred_cam_count = (
        #     self.info_room.cam_extrinsics[:, :, :, 0].sum().type(self.INT).item()
        # )
        # vis_mask = None

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
            if _is_in(pos, candidates):
                direction_similarity, max_delta_cov = (
                    self.info_room.direction_similarity(
                        pos, direction, section_nums, cam_type
                    )
                )
                lov_indices, lov_check_list = if_need_obstacle_check(
                    pos, self.info_room.must_monitor, self.info_room.occupancy
                )
                self.info_room.add_cam(
                    pos, direction, cam_type, lov_indices, lov_check_list
                )
            else:
                # import pdb; pdb.set_trace()
                action = ILLEGAL
        else:
            raise ValueError(f"Unknown action {action}.")

        cur_coverage: float = (
            self.info_room.visible_points > 0
        ).sum().item() / total_target_point_num

        # if abs(cur_coverage - pred_coverage) <= 0.01:
        #     self.lazy_count += 1
        # else:
        #     self.lazy_count = 0

        cam_count = (
            self.info_room.cam_extrinsics[:, :, :, 0].sum().type(self.INT).item()
        )

        if action == ILLEGAL:
            # import pdb; pdb.set_trace()
            raise RuntimeError("ILLEGAL action operated.")

        reward = 0
        cov_incre = cur_coverage - pred_coverage

        # ==== 1 ====
        # if max_delta_cov == 0:
        #     reward += -100
        # elif cov_incre == 0:
        #     reward += max_delta_cov * direction_similarity * 100
        # else:  # cov_incre > 0
        #     reward += (1 + cov_incre) * 100

        # ==== 2 ====
        if max_delta_cov == 0:
            reward += -200
        elif cov_incre == 0:
            reward += (max_delta_cov * direction_similarity - 1) * 100
        else:  # cov_incre > 0
            reward += cov_incre * 100

        if cur_coverage == 1:
            reward += 200
        if cov_incre > 0.05:
            reward += cov_incre // 0.05 * 10

        terminated = (
            action == STOP or self.action_count == total_steps or cur_coverage == 1
        )
        transition = dict(
            next_observation=self.info_room.get_info(),
            reward=reward,
            terminated=terminated,
        )

        if terminated:
            self.lazy_count = 0

        return transition, cur_coverage, cam_count


def _is_in(pos: array, candidates: array):
    extend_pos = np.tile(pos, [len(candidates), 1])
    return np.any(np.all(extend_pos == candidates, axis=1), axis=0)
