from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from labsurv.builders import ENVIRONMENTS
from labsurv.models.envs import BaseSurveillanceEnv
from labsurv.utils.surveillance import (
    if_need_obstacle_check,
    pack_observation2transition,
)
from numpy import ndarray as array


@ENVIRONMENTS.register_module()
class OCPPPOEnv(BaseSurveillanceEnv):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        room_data_path: str,
        device: Optional[str],
        save_path: str,
        reward_goals: List[float] = [1.0],
        reward_bonus: List[List[float]] = [[0.0, 0.005]],
        terminate_goal: float = 1.0,
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
            timestep. When a camera is installed, the coverage will increase or
            decrease accordingly, which gives the reward of the action at current
            state.

        ## Start state:

            No camera is installed at the very beginning.

        ## Episode End

            The episode truncates if the episode length is greater than 50.
        """

        super().__init__(room_data_path, device, save_path)

        self.reward_goals = reward_goals
        self.reward_bonus = reward_bonus
        self.terminate_goal = terminate_goal

    def step(
        self,
        observation: array,
        action_with_params: array,
        total_steps: int,
    ) -> Tuple[float, Dict[str, bool | float | array]]:
        """
        ## Description:

            Run one timestep of the environment's dynamics.

        ## Arguments:

            observation (np.ndarray): [1, 2, W, D, H], np.float32, the policy input array.

            action_with_params (np.ndarray): [9].
        """
        ADD = 0
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
        cur_coverage: float = -1

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
                lov_indices, lov_check_list = if_need_obstacle_check(
                    pos, self.info_room.must_monitor, self.info_room.occupancy
                )
                self.info_room.add_cam(
                    pos, direction, cam_type, lov_indices, lov_check_list
                )
                cur_coverage: float = (
                    self.info_room.visible_points > 0
                ).sum().item() / total_target_point_num
            else:
                # import pdb; pdb.set_trace()
                raise RuntimeError("ILLEGAL action operated.")
        else:
            raise ValueError(f"Unknown action {action}.")

        output_transition = dict(
            cur_observation=observation.squeeze(0),  # [2, W, D, H]
            cur_action=action_with_params,  # [9]
            next_observation=pack_observation2transition(
                self.info_room.get_info()
            ).squeeze(
                0
            ),  # [2, W, D, H]
            reward=_compute_reward(
                cur_coverage,
                pred_coverage,
                self.reward_goals,
                self.reward_bonus,
                self.terminate_goal,
            ),
            terminated=(
                self.action_count == total_steps or cur_coverage >= self.terminate_goal
            ),
        )

        return cur_coverage, output_transition


def _is_in(pos: array, candidates: array):
    extend_pos = np.tile(pos, [len(candidates), 1])
    return np.any(np.all(extend_pos == candidates, axis=1), axis=0)


def _compute_reward(
    cur_coverage: float,
    pred_coverage: float,
    goal_coverages: List[float] = [0.25, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    bonus_incres: List[List[float]] = [
        [0.0, 0.05],
        [0.6, 0.02],
        [0.8, 0.005],
        [0.9, 0.001],
    ],
    terminated_coverage: float = 1.0,
) -> float:
    reward: float = 0
    cov_incre = cur_coverage - pred_coverage

    if cov_incre == 0:
        return -5

    reward += cov_incre * 100

    goal_coverages = sorted(goal_coverages)
    for goal in goal_coverages:
        if (cur_coverage >= goal and pred_coverage < goal) or (
            cur_coverage > goal and pred_coverage <= goal
        ):
            reward += 50

    bonus_incres.append([terminated_coverage, 1])
    bonus_incres = sorted(bonus_incres, key=lambda x: x[0])
    for index, item in enumerate(bonus_incres):
        lower_bound, incre_step = item
        if lower_bound == terminated_coverage:
            break

        next_lower_bound = bonus_incres[index + 1][0]
        if cur_coverage > lower_bound and pred_coverage < next_lower_bound:
            reward += (
                (min(cur_coverage, next_lower_bound) - max(lower_bound, pred_coverage))
                // incre_step
                * 5
            )

    return reward
