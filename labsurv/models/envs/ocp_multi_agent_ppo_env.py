import os
import os.path as osp
import pickle
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from labsurv.builders import ENVIRONMENTS
from labsurv.models.envs import BaseSurveillanceEnv
from labsurv.physics import SurveillanceRoom
from labsurv.utils.surveillance import apply_movement_on_agent, if_need_obstacle_check
from mmcv.utils import ProgressBar
from numpy import ndarray as array
from numpy import pi as PI
from torch import Tensor


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
        reset_count_path: Optional[str] = None,
        pan_section_num: int = 360,
        tilt_section_num: int = 180,
        pan_range: List[float] = [-PI, PI],
        tilt_range: List[float] = [-PI / 2, PI / 2],
        cam_types: int = 1,
        terminate_goal: float = 1.0,
        reset_weight: int = 4,
        vismask_path: Optional[str] = "",
        cache_vismask: bool = False,
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

        self.terminate_goal = terminate_goal
        self.reset_weight = reset_weight
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
        self.cache_vismask = cache_vismask
        self._load_reset_pos_count(reset_count_path)
        self._load_vismask(vismask_path)

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

    def _load_vismask(self, pkl_path: Optional[str] = None):
        self.use_cache_vismask = True
        if pkl_path is not None:
            if pkl_path != "":
                with open(pkl_path, "rb") as f:
                    self.vismasks = pickle.load(f)
            else:
                self.use_cache_vismask = False
        elif not self.cache_vismask:
            self.vismasks = _build_vismask(
                self._surv_room,
                self.pos_candidates,
                self.pan_section_num,
                self.tilt_section_num,
                self.pan_range,
                self.tilt_range,
                self.cam_types,
            )

            with open(osp.join(self.save_path, "vismasks.pkl"), "wb") as f:
                pickle.dump(self.vismasks, f)
        else:
            self.vismasks = _build_vismask(
                None,
                self.pos_candidates,
                self.pan_section_num,
                self.tilt_section_num,
                self.pan_range,
                self.tilt_range,
                self.cam_types,
            )

    def reset(self, seed: Optional[int] = None) -> Tuple[array, array]:
        """
        ## Description:

            Reset the environment to the initial state, randomly install a number of
            `self.agent_num` cameras and return the env info.

        ## Arguments:

            seed (Optional[int])

        ## Returns:

            info (np.ndarray): [12, W, D, H], merged by all the tensor attributes of
            the room.

            cam_params (np.ndarray): [AGENT_NUM, 6], the randomly installed camera
            params.
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

        # randomly generate cameras and install
        upper_count = np.max(self.visit_count) + 1
        position_indices: array = self._np_random.choice(
            len(self.pos_candidates),
            self.agent_num,
            replace=False,
            p=(
                (upper_count - self.visit_count) ** self.reset_weight
                / np.sum((upper_count - self.visit_count) ** self.reset_weight)
            ),
        )  # [AGENT_NUM, 1]
        positions = []
        for index in position_indices:
            positions.append(self.pos_candidates[index])
            self.visit_count[index] += 1
        positions = np.array(positions, dtype=np.int64)

        # DEBUG(eric)
        # positions: array = np.array([[4, 5, 17]], dtype=np.int64)

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
                    for tilt_index in range(1, self.tilt_section_num)
                ]
            ]
        ).transpose()
        directions: array = np.concatenate(
            (
                self._np_random.choice(pan_candidates, self.agent_num),
                self._np_random.choice(tilt_candidates, self.agent_num),
            ),
            axis=1,
        )  # [AGENT_NUM, 2]

        cam_types: array = self._np_random.choice(
            np.array([[i for i in range(self.cam_types)]]).transpose(),
            self.agent_num,
        ).astype(np.int64)

        print("Resetting environment...")
        prog_bar = ProgressBar(self.agent_num)
        for param_index in range(self.agent_num):
            lov_indices, lov_check_list = if_need_obstacle_check(
                positions[param_index],
                self.info_room.must_monitor,
                self.info_room.occupancy,
            )
            self.info_room.add_cam(
                positions[param_index],
                directions[param_index],
                cam_types[param_index][0],
                lov_indices,
                lov_check_list,
            )

            prog_bar.update()
        print("\r\033[K\033[1A\033[K\033[1A", end="")

        # clear modification history for random generation
        self.info_room.cam_modify_num = 0
        cam_params: array = np.concatenate(
            (positions, directions, cam_types), axis=1
        )  # [AGENT_NUM, 6]

        return self.info_room.get_info(), cam_params

    def step(
        self,
        observation: array,
        action: array,
        total_steps: int,
    ) -> Tuple[float, Dict[str, bool | float | array]]:
        """
        ## Description:

            Run one timestep of the environment's dynamics.

        ## Arguments:

            observation (np.ndarray): [1, 6], np.float32.

            action (np.ndarray): [7], np.int64
        """
        self.action_count += 1

        cur_pos: array = observation[0, :3].copy()  # [3]
        cur_direction: array = observation[0, 3:5].copy()  # [2]
        cur_cam_type: int = int(observation[0, 5])
        cur_action: array = action[:6].copy()
        new_observation: array = apply_movement_on_agent(  # [6]
            observation.squeeze(0),
            cur_action,
            self.info_room.shape,
            self.pan_section_num,
            self.tilt_section_num,
            self.pan_range,
            self.tilt_range,
        )
        new_pos: array = new_observation[:3].copy()  # [3]
        new_direction: array = new_observation[3:5].copy()  # [2]
        new_cam_type: int = int(new_observation[5])

        total_target_point_num: float = (
            self.info_room.must_monitor[:, :, :, 0].sum().item()
        )
        pred_coverage: float = (
            self.info_room.visible_points > 0
        ).sum().item() / total_target_point_num
        cur_coverage: float = -1

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
        print("\nModifying cameras...", end="")
        cur_lov_indices, cur_lov_check_list = if_need_obstacle_check(
            cur_pos, self.info_room.must_monitor, self.info_room.occupancy
        )
        print("\r\033[K\033[1A", end="")

        pan_step: float = (self.pan_range[1] - self.pan_range[0]) / self.pan_section_num
        tilt_step: float = (
            self.tilt_range[1] - self.tilt_range[0]
        ) / self.tilt_section_num
        cur_pan_index: int = round((cur_direction[0] - self.pan_range[0]) / pan_step)
        cur_tilt_index: int = round(
            (cur_direction[1] - self.tilt_range[0]) / tilt_step - 1
        )
        new_pan_index: int = round((new_direction[0] - self.pan_range[0]) / pan_step)
        new_tilt_index: int = round(
            (new_direction[1] - self.tilt_range[0]) / tilt_step - 1
        )

        if _is_in(new_pos, candidates):
            if self.use_cache_vismask:
                provided_pred_vismask = self.vismasks[tuple(cur_pos.astype(np.int64))][
                    cur_pan_index
                ][cur_tilt_index][cur_cam_type]
                provided_vismask = self.vismasks[tuple(new_pos.astype(np.int64))][
                    new_pan_index
                ][new_tilt_index][new_cam_type]

                if provided_pred_vismask is None or provided_vismask is None:
                    print(
                        "\nChanging camera position: step 1 | Deleting camera...",
                        end="",
                    )
                    new_lov_indices, new_lov_check_list = if_need_obstacle_check(
                        new_pos, self.info_room.must_monitor, self.info_room.occupancy
                    )

                    provided_pred_vismask = self.info_room.del_cam(
                        cur_pos, cur_lov_indices, cur_lov_check_list
                    )
                    self.vismasks[tuple(cur_pos.astype(np.int64))][cur_pan_index][
                        cur_tilt_index
                    ][cur_cam_type] = torch.tensor(
                        provided_pred_vismask, dtype=torch.int64, device=self.device
                    )
                    print("\r\033[K\033[1A", end="")
                    print(
                        "\nChanging camera position: step 2 | Adding new camera...",
                        end="",
                    )
                    provided_vismask = self.info_room.add_cam(
                        new_pos,
                        new_direction,
                        new_cam_type,
                        new_lov_indices,
                        new_lov_check_list,
                    )
                    self.vismasks[tuple(new_pos.astype(np.int64))][new_pan_index][
                        new_tilt_index
                    ][new_cam_type] = torch.tensor(
                        provided_vismask, dtype=torch.int64, device=self.device
                    )
                    print("\r\033[K\033[1A", end="")
                else:
                    print("\nChanging camera position...")
                    self.info_room.del_cam(
                        cur_pos,
                        provided_vismask=provided_pred_vismask,
                    )
                    self.info_room.add_cam(
                        new_pos,
                        new_direction,
                        new_cam_type,
                        provided_vismask=provided_vismask,
                    )
                    print("\r\033[K\033[1A\033[K", end="")
            else:
                print("\nChanging camera position: step 1 | Deleting camera...", end="")
                new_lov_indices, new_lov_check_list = if_need_obstacle_check(
                    new_pos, self.info_room.must_monitor, self.info_room.occupancy
                )
                self.info_room.del_cam(cur_pos, cur_lov_indices, cur_lov_check_list)
                print("\r\033[K\033[1A\033[K\033[1A", end="")

                print(
                    "\nChanging camera position: step 2 | Adding new camera...", end=""
                )
                provided_vismask = self.info_room.add_cam(
                    new_pos,
                    new_direction,
                    new_cam_type,
                    new_lov_indices,
                    new_lov_check_list,
                )
                print("\r\033[K\033[1A\033[K\033[1A", end="")
        else:
            new_observation[:3] = cur_pos
            new_pos = cur_pos.copy()

            if self.use_cache_vismask:
                provided_pred_vismask = self.vismasks[tuple(cur_pos.astype(np.int64))][
                    cur_pan_index
                ][cur_tilt_index][cur_cam_type]
                provided_vismask = self.vismasks[tuple(cur_pos.astype(np.int64))][
                    new_pan_index
                ][new_tilt_index][new_cam_type]

                if provided_pred_vismask is None or provided_vismask is None:
                    print("\nChanging camera direction and/or type...")
                    (
                        provided_pred_vismask,
                        provided_vismask,
                    ) = self.info_room.adjust_cam(
                        cur_pos,
                        new_direction,
                        new_cam_type,
                        cur_lov_indices,
                        cur_lov_check_list,
                        delta_vismask_out=False,
                    )

                    self.vismasks[tuple(cur_pos.astype(np.int64))][cur_pan_index][
                        cur_tilt_index
                    ][cur_cam_type] = torch.tensor(
                        provided_pred_vismask, dtype=torch.int64, device=self.device
                    )
                    self.vismasks[tuple(new_pos.astype(np.int64))][new_pan_index][
                        new_tilt_index
                    ][new_cam_type] = torch.tensor(
                        provided_vismask, dtype=torch.int64, device=self.device
                    )
                    print("\r\033[K\033[1A\033[K\033[1A", end="")
                else:
                    print("\nChanging camera direction and/or type...")
                    self.info_room.adjust_cam(
                        cur_pos,
                        new_direction,
                        new_cam_type,
                        provided_pred_vismask=provided_pred_vismask,
                        provided_vismask=provided_vismask,
                    )
                    print("\r\033[K\033[1A\033[K", end="")
            else:
                print("\nChanging camera direction and/or type...")
                self.info_room.adjust_cam(
                    cur_pos,
                    new_direction,
                    new_cam_type,
                    cur_lov_indices,
                    cur_lov_check_list,
                    delta_vismask_out=False,
                )
                print("\r\033[K\033[1A\033[K\033[1A", end="")

        new_pos_ind = np.nonzero(
            np.all(
                np.tile(new_pos, [len(self.pos_candidates), 1]) == self.pos_candidates,
                axis=1,
            )
        )
        self.visit_count[new_pos_ind] += 1

        cur_coverage: float = (
            self.info_room.visible_points > 0
        ).sum().item() / total_target_point_num

        if np.abs(np.sum(observation.squeeze(0) - new_observation)) < 1e-5:
            self.lazy_count += 1
        else:
            self.lazy_count = 0

        output_transition = dict(
            cur_observation=observation.squeeze(0),  # [6]
            cur_action=action,  # [7]
            next_observation=new_observation,  # [6]
            reward=_compute_reward(
                cur_coverage,
                pred_coverage,
                self.terminate_goal,
                self.lazy_count,
            ),
            terminated=(
                self.action_count == total_steps or cur_coverage >= self.terminate_goal
            ),
        )

        return cur_coverage, output_transition

    def save(self, episode_index: int, save_path: str):
        episode = episode_index + 1
        if save_path.endswith(".pkl"):
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            env_save_path = (
                ".".join(save_path.split(".")[:-1]) + f"_episode_{episode}.pkl"
            )
            vismask_save_path = ".".join(save_path.split(".")[:-1]) + ".pkl"
        else:
            os.makedirs(save_path, exist_ok=True)
            env_save_path = osp.join(save_path, f"visit_count_episode_{episode}.pkl")
            vismask_save_path = osp.join(save_path, "vismasks.pkl")

        dump_dict = dict(
            visit_count=self.visit_count,
            candidates=self.pos_candidates,
        )
        with open(env_save_path, "wb") as f:
            pickle.dump(dump_dict, f)

        if self.use_cache_vismask:
            with open(vismask_save_path, "wb") as f:
                pickle.dump(self.vismasks, f)


def _build_vismask(
    room: Optional[SurveillanceRoom],
    pos_candidates: array,
    pan_section_num: int,
    tilt_section_num: int,
    pan_range: List[float],
    tilt_range: List[float],
    cam_types: int,
) -> Dict[Tuple[int], Dict[int, Dict[int, Dict[int, Tensor]]]]:
    """
    ## Returns:

        vismasks (Dict[int, Dict[int, Dict[int, array]]]):
        Index in the format of vismasks[pan_index][tilt_index][cam_type_index].
    """

    pan_step: float = (pan_range[1] - pan_range[0]) / pan_section_num
    pan_list: List[float] = [
        pan_range[0] + i * pan_step for i in range(pan_section_num)
    ]
    tilt_step: float = (tilt_range[1] - tilt_range[0]) / tilt_section_num
    tilt_list: List[float] = [
        tilt_range[0] + i * tilt_step for i in range(1, tilt_section_num)
    ]

    # vismasks[pos][pan_index][tilt_index][cam_type_index]
    vismasks: Dict[Tuple[int], Dict[int, Dict[int, Dict[int, Tensor]]]] = {
        tuple(pos): {
            pan_index: {
                tilt_index: {cam_type: None for cam_type in range(cam_types)}
                for tilt_index in range(len(tilt_list))
            }
            for pan_index in range(len(pan_list))
        }
        for pos in pos_candidates
    }

    if room is not None:
        room = deepcopy(room)

        print("\nBuilding speed up indices for positions...")
        pos_prog = ProgressBar(len(pos_candidates))
        for pos in pos_candidates:
            lov_indices, lov_check_list = if_need_obstacle_check(
                pos, room.must_monitor, room.occupancy
            )

            print("\nBuilding speed up indices for directions...")
            ang_prog = ProgressBar(len(pan_list) * len(tilt_list))
            for pan_index, pan in enumerate(pan_list):
                for tilt_index, tilt in enumerate(tilt_list):
                    print("\nBuilding speed up indices for cam_types...")
                    cam_type_prog = ProgressBar(cam_types)
                    for cam_type in range(cam_types):
                        direction: array = np.array([pan, tilt])

                        cache_room = deepcopy(room)
                        vismasks[tuple(pos)][pan_index][tilt_index][cam_type] = (
                            torch.tensor(
                                cache_room.add_cam(
                                    pos,
                                    direction,
                                    cam_type,
                                    lov_indices,
                                    lov_check_list,
                                ),
                                dtype=torch.int64,
                                device=room.device,
                                requires_grad=False,
                            )
                        )
                        cam_type_prog.update()
                    print("\r\033[K\033[1A\033[K\033[1A", end="")

                    ang_prog.update()
            print("\r\033[K\033[1A\033[K\033[1A", end="")

            pos_prog.update()
        print("\r\033[K\033[1A\033[K\033[1A", end="")

    return vismasks


def _is_in(pos: array, candidates: array):
    extend_pos = np.tile(pos, [len(candidates), 1])
    return np.any(np.all(extend_pos == candidates, axis=1), axis=0)


def _compute_reward(
    cur_coverage: float,
    pred_coverage: float,
    terminated_coverage: float = 1.0,
    lazy_count: int = 0,
) -> float:
    reward: float = 0

    if cur_coverage >= terminated_coverage:
        reward += 50
    else:
        reward += -1  # np.sign(cur_coverage - pred_coverage) if lazy_count == 0 else -1

    return reward
