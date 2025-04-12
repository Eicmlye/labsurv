import os
import os.path as osp
import pickle
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from labsurv.builders import ENVIRONMENTS
from labsurv.models.envs import BaseSurveillanceEnv
from labsurv.runners import LoggerHook
from labsurv.utils.surveillance import (
    apply_movement_on_agent,
    array_is_in,
    info_room2actor_input,
    info_room2critic_input,
)
from mmcv.utils import ProgressBar
from numpy import ndarray as array
from numpy import pi as PI


@ENVIRONMENTS.register_module()
class OCPMultiAgentPPOEnv(BaseSurveillanceEnv):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        room_data_path: str,
        device: Optional[str],
        save_path: str,
        agent_num: int,
        visit_count_path: Optional[str] = None,
        pan_section_num: int = 360,
        tilt_section_num: int = 180,
        pan_range: List[float] = [-PI, PI],
        tilt_range: List[float] = [-PI / 2, PI / 2],
        cam_types: int = 1,
        reset_rand_prob: float = 0.5,
        reset_pos: str = "center",  # start, center
        subgoals: List[List[float]] = [[0, 0]],
        terminate_goal: float = 1.0,
        reset_weight: int = 4,
        individual_reward_alpha: float = 0,
        allow_polar: bool = False,
    ):
        """
        ## Description:

            This environment is the MAPPO surveillance room environment class.

        ## Action space:

            The action is an `np.ndarray` shaped [6], [delta_x, delta_y, delta_z,
            delta_pan, delta_tilt, cam_type_choice].

            For positional coords, +1 or -1 is allowed.

            For pan-tilt coords, consider `pan` and `tilt` as the axis of the angular
            space. The coords moves on the grid points, and polar points are ignored.

            For cam_type action, it is treated as a choice of camera.

        ## Observation space:

            The agent observes its own position, direction and cam_type.

        ## Rewards:

            Single agent rewards are simply represented by the coverage increment at
            current timestep.

        ## Start state:

            A certain number of cameras are randomly installed in the environment.

        ## Episode End:

            The episode truncates if the episode length is greater than 50.
        """

        super().__init__(room_data_path, device, save_path)

        self.agent_num: int = agent_num

        self.pan_section_num = pan_section_num
        self.tilt_section_num = tilt_section_num
        self.pan_range = pan_range
        self.tilt_range = tilt_range
        self.cam_types = cam_types
        self.allow_polar = allow_polar

        self.reset_rand_prob = reset_rand_prob
        self.reset_pos = reset_pos
        if self.reset_pos not in ["start", "center"]:
            raise NotImplementedError()
        self.subgoals = subgoals
        self.terminate_goal = terminate_goal
        self.reset_weight = reset_weight
        self.individual_reward_alpha = individual_reward_alpha
        self.pos_candidates: array = (
            (
                torch.logical_xor(
                    self._surv_room.install_permitted,
                    self._surv_room.cam_extrinsics[:, :, :, 0],
                )
                .nonzero()
                .type(self.INT)
            )
            .cpu()
            .numpy()
            .copy()
        )
        self._load_reset_pos_count(visit_count_path)

    def _load_reset_pos_count(self, pkl_path: Optional[str] = None):
        if pkl_path is None:
            self.visit_count: array = np.zeros(
                (len(self.pos_candidates),), dtype=np.int64
            )
        else:
            with open(pkl_path, "rb") as f:
                self.visit_count: array = pickle.load(f)["visit_count"]
            if len(self.pos_candidates) != len(self.visit_count):
                raise ValueError("Loaded file does not match `_surv_room`.")

    def reset(
        self, seed: Optional[int] = None, rand_init: Optional[bool] = True, **kwargs
    ) -> Tuple[array, array]:
        """
        ## Description:

            Reset the environment to the initial state, randomly install a number of
            `self.agent_num` cameras and return the env info.

        ## Arguments:

            seed (Optional[int])

            rand_init (Optional[bool]): `True` for always random, `False` for always
            specified, `None` for random under `self.reset_rand_prob`.

        ## Returns:

            room (SurveillanceRoom): deepcopy of `info_room`.

            cam_params (np.ndarray): [AGENT_NUM, PARAM_DIM], the randomly installed
            camera params.
        """
        logger: LoggerHook = kwargs["logger"]

        # do env init works
        super().reset(seed=seed)

        # return init observation according to observation distribution
        del self.info_room
        self.info_room = deepcopy(self._surv_room)

        self.action_count = 0
        self.lazy_count = 0

        if rand_init is True or (
            rand_init is None and random.uniform(0, 1) < self.reset_rand_prob
        ):
            logger.show_log("Reset with random params.")
            rand_operation = True
        else:
            logger.show_log("Reset with specified params.")
            rand_operation = False

        # randomly generate cameras and install
        upper_count = np.max(self.visit_count) + 1
        if rand_operation:
            try:
                visit_probs = (
                    upper_count - self.visit_count
                ) ** self.reset_weight / np.sum(
                    (upper_count - self.visit_count) ** self.reset_weight
                )
                position_indices: array = self._np_random.choice(
                    len(self.pos_candidates),
                    self.agent_num,
                    replace=False,
                    p=visit_probs,
                )  # [AGENT_NUM, 1]
            except ValueError:  # probability not non-negative
                neg_list = [
                    (visit_probs[ind], ind)
                    for ind in range(len(visit_probs))
                    if visit_probs[ind] < 0
                ]
                logger.show_log(
                    f"[WARN]  negative probability for {neg_list}"
                    "\nReset with center params."
                )
        else:
            if self.reset_pos == "start":
                position_indices: array = np.array([[i] for i in range(self.agent_num)])
            elif self.reset_pos == "center":
                position_indices: array = np.array(
                    [
                        [(len(self.pos_candidates) - self.agent_num) // 2 + i]
                        for i in range(self.agent_num)
                    ]
                )
            else:
                raise NotImplementedError()
        positions = []
        for index in position_indices:
            positions.append(self.pos_candidates[index])
            self.visit_count[index] += 1
        positions = np.array(positions, dtype=np.int64).reshape(-1, 3)  # [AGENT_NUM, 3]

        pan_step = (self.pan_range[1] - self.pan_range[0]) / self.pan_section_num
        tilt_step = (self.tilt_range[1] - self.tilt_range[0]) / self.tilt_section_num
        pan_candidates: array = np.array(
            [
                [
                    self.pan_range[0] + pan_index * pan_step
                    for pan_index in range(self.pan_section_num)
                ]
            ]
        ).transpose()
        tilt_candidates: array = np.array(
            [
                [
                    self.tilt_range[0] + tilt_index * tilt_step
                    for tilt_index in range(
                        0 if self.allow_polar else 1,
                        (
                            self.tilt_section_num + 1
                            if self.allow_polar
                            else self.tilt_section_num
                        ),
                    )
                ]
            ]
        ).transpose()
        if rand_operation:
            pan: array = self._np_random.choice(pan_candidates, self.agent_num)
            tilt: array = self._np_random.choice(tilt_candidates, self.agent_num)
        else:
            if self.reset_pos == "start":
                pan_indices: array = np.array([[0] for i in range(self.agent_num)])
                tilt_indices: array = np.array([[0] for i in range(self.agent_num)])
            elif self.reset_pos == "center":
                pan_indices: array = np.array(
                    [[len(pan_candidates) // 2] for i in range(self.agent_num)]
                )
                tilt_indices: array = np.array(
                    [[len(tilt_candidates) // 2] for i in range(self.agent_num)]
                )
            else:
                raise NotImplementedError()

            pan_list = []
            tilt_list = []
            for pan_index, tilt_index in zip(pan_indices, tilt_indices):
                pan_list.append(pan_candidates[pan_index])
                tilt_list.append(tilt_candidates[tilt_index])

            pan: array = np.array(pan_list).reshape(-1, 1)
            tilt: array = np.array(tilt_list).reshape(-1, 1)
        directions: array = np.concatenate((pan, tilt), axis=1)  # [AGENT_NUM, 2]

        cam_type_indices: array = (
            self._np_random.choice(  # [AGENT_NUM]
                np.array(
                    [[i if rand_operation else 0 for i in range(self.cam_types)]]
                ).transpose(),
                self.agent_num,
            )
            .astype(np.int64)
            .reshape(-1)
        )
        # [AGENT_NUM, CAM_TYPES]
        one_hot_cam_types: array = np.eye(self.cam_types)[cam_type_indices]

        print("Resetting environment...")
        prog_bar = ProgressBar(self.agent_num)
        for param_index in range(self.agent_num):
            self.info_room.add_cam(
                positions[param_index],
                directions[param_index],
                cam_type_indices[param_index],
            )

            prog_bar.update()
        print("\r\033[K\033[1A\033[K\033[1A", end="")

        # clear modification history for random generation
        self.info_room.cam_modify_num = 0
        cam_params: array = np.concatenate(
            (positions, directions, one_hot_cam_types), axis=1
        )  # [AGENT_NUM, PARAM_DIM]

        return deepcopy(self.info_room), cam_params

    def step(
        self,
        observations: List[Tuple[array, array, array]],
        actions: List[array],
        total_steps: int,
        action_masks: Optional[List[array]] = None,
    ) -> Tuple[float, Dict[str, bool | float | array]]:
        """
        ## Description:

            Run one timestep of the environment's dynamics.

        ## Arguments:

            observations (List[Tuple[array, array, array]]): [AGENT_NUM, Tuple], actor
            inputs.

            actions (List[array]): [AGENT_NUM, ACTION_DIM].
        """
        cur_critic_input = info_room2critic_input(self.info_room, self.agent_num)
        self.action_count += 1

        pred_coverage: float = self.info_room.coverage
        pred_vis_count: array = self.info_room.visible_points.cpu().numpy().copy()

        new_params: List[array] = []
        new_vismasks: List[array] = []
        pred_vismasks: List[array] = []

        for agent_index in range(self.agent_num):
            observation: Tuple[array, array, array] = observations[agent_index]
            action: array = actions[agent_index]
            params: array = observation[0]  # [AGENT_NUM(NEIGH), PARAM_DIM]
            self_mask: array = observation[1]  # [AGENT_NUM(NEIGH)]
            cur_param: array = params[self_mask].reshape(-1)  # [PARAM_DIM]

            cur_pos: array = cur_param[:3].copy()  # [3]
            new_param: array = apply_movement_on_agent(  # [PARAM_DIM]
                cur_param,
                action,
                self.info_room.shape,
                self.pan_section_num,
                self.tilt_section_num,
                self.pan_range,
                self.tilt_range,
                allow_polar=self.allow_polar,
            )
            new_pos: array = new_param[:3].copy()  # [3]
            new_direction: array = new_param[3:5].copy()  # [2]
            new_cam_type: int = round(np.array(new_param[5:].nonzero()).reshape(-1)[0])

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
            new_vismask: array = None
            pred_vismask: array = None

            print(f"\nEditing agent {agent_index + 1}/{self.agent_num}...")
            if array_is_in(new_pos, candidates):
                pred_vismask = self.info_room.del_cam(cur_pos)
                new_vismask = self.info_room.add_cam(
                    new_pos, new_direction, new_cam_type
                )
            else:
                new_param[:3] = cur_pos
                new_pos = cur_pos.copy()
                pred_vismask, new_vismask = self.info_room.adjust_cam(
                    cur_pos,
                    new_direction,
                    new_cam_type,
                    delta_vismask_out=False,
                )
            print("\r\033[K\033[1A\033[K\033[1A", end="")

            new_params.append(new_param)
            new_vismasks.append(new_vismask)
            pred_vismasks.append(pred_vismask)

            new_pos_ind = np.nonzero(
                np.all(
                    np.tile(new_pos, [len(self.pos_candidates), 1])
                    == self.pos_candidates,
                    axis=1,
                )
            )
            self.visit_count[new_pos_ind] += 1

        cur_coverage: float = self.info_room.coverage

        next_observations = []
        for agent_index, param in enumerate(new_params):
            next_observations.append(
                info_room2actor_input(self.info_room, self.agent_num, param)
            )

        output_transition = dict(
            cur_observation=observations,  # [AGENT_NUM, Tuple]
            cur_action=actions,  # [AGENT_NUM, ACTION_DIM]
            cur_action_mask=action_masks,  # [AGENT_NUM, ACTION_DIM]
            cur_critic_input=cur_critic_input,
            next_observation=next_observations,  # [AGENT_NUM, Tuple]
            next_critic_input=info_room2critic_input(self.info_room, self.agent_num),
            reward=self.compute_reward(  # [AGENT_NUM + 1]
                cur_coverage,
                pred_coverage,
                pred_vismasks,
                pred_vis_count,
                new_vismasks,
                self.info_room.visible_points.cpu().numpy().copy(),
                self.info_room.must_monitor[:, :, :, 0].sum().item(),
            ),
            terminated=(
                self.action_count == total_steps or cur_coverage >= self.terminate_goal
            ),
        )

        return cur_coverage, output_transition, np.array(new_params, dtype=np.float32)

    def save(self, episode_index: int, save_path: str):
        episode = episode_index + 1
        if save_path.endswith(".pkl"):
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            env_save_path = (
                ".".join(save_path.split(".")[:-1]) + f"_episode_{episode}.pkl"
            )
        else:
            os.makedirs(save_path, exist_ok=True)
            env_save_path = osp.join(save_path, f"visit_count_episode_{episode}.pkl")

        dump_dict = dict(
            visit_count=self.visit_count,
            candidates=self.pos_candidates,
        )
        with open(env_save_path, "wb") as f:
            pickle.dump(dump_dict, f)

    def compute_reward(
        self,
        cur_coverage: float,
        pred_coverage: float,
        pred_vismasks: List[array],
        pred_vis_count: array,
        new_vismasks: List[array],
        new_vis_count: array,
        total_target_num: int,
    ) -> List[float]:
        assert len(pred_vismasks) == len(new_vismasks)

        individual_rewards = []
        cov_delta = cur_coverage - pred_coverage

        total_reward: float = cov_delta
        if cur_coverage >= self.terminate_goal:
            total_reward += self.terminate_goal

        # subgoals
        for goal, bonus in self.subgoals:
            if pred_coverage < goal and cur_coverage >= goal:
                total_reward += bonus
            elif pred_coverage >= goal and cur_coverage < goal:
                total_reward -= bonus

        # agent-wise delta coverage reward credit
        if self.individual_reward_alpha > 0:
            for mask_index in range(len(new_vismasks)):
                agent_new_cov: float = (
                    np.sum(1 / new_vis_count[new_vismasks[mask_index] > 0])
                    if len(new_vis_count[new_vismasks[mask_index] > 0]) > 0
                    else 0.0
                ) / total_target_num
                agent_pred_cov: float = (
                    np.sum(1 / pred_vis_count[pred_vismasks[mask_index] > 0])
                    if len(pred_vis_count[pred_vismasks[mask_index] > 0]) > 0
                    else 0.0
                ) / total_target_num

                individual_rewards.append(agent_new_cov - agent_pred_cov)

            mixed_rewards = (
                self.individual_reward_alpha * np.array(individual_rewards)
                + (1 - self.individual_reward_alpha) * total_reward
            )
        else:
            mixed_rewards = np.zeros((self.agent_num,), dtype=np.float32)

        rewards = mixed_rewards.tolist() + [total_reward]

        return rewards
