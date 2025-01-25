from typing import Dict, List, Optional

import numpy as np
import torch
from labsurv.builders import ENVIRONMENTS
from numpy import ndarray as array


@ENVIRONMENTS.register_module()
class CliffWalkDQNEnv:
    INT = torch.int64
    FLOAT = torch.float

    FREE = 0
    FROM = 1
    DEAD = 2
    DEST = 3

    def __init__(
        self,
        env: List[int],
        device: Optional[str],
        save_path: str = None,
    ):
        self.device = torch.device(device)
        self.env: array = np.array(env, dtype=np.int64)
        assert (
            len(np.transpose(np.nonzero(self.env == self.FROM))) == 1
        ), "Multiple start points found."
        self.env_count: array = np.zeros_like(self.env)

    @property
    def env_shape(self):
        return list(self.env.shape)

    @property
    def start_point_index(self):
        coord = np.transpose(np.nonzero(self.env == self.FROM))[0]

        obs_index = _coord2obs_index(coord, self.env_shape)

        return obs_index

    def reset(self, test_mode: bool = False) -> array:
        if test_mode:
            return self.start_point_index

        coords = np.transpose(np.nonzero(self.env != self.DEAD))

        obs_index = _coord2obs_index(
            coords[np.random.choice(len(coords))], self.env_shape
        )

        return obs_index

    def step(
        self,
        observation: array,
        action: array,
    ) -> Dict[str, array | int | bool]:
        cur_obs_coord = _obs_index2_coord(observation, self.env_shape)
        self.env_count[cur_obs_coord[0], cur_obs_coord[1]] += 1
        movement = _action_index2movement(action, 2)

        env_shape: array = np.array(self.env_shape, dtype=np.int64)
        upper_bound: array = env_shape - 1
        lower_bound: array = np.zeros_like(upper_bound, dtype=np.int64)

        next_obs_coord: array = np.clip(
            cur_obs_coord + movement, lower_bound, upper_bound, dtype=np.int64
        )
        next_pos_state = self.env[next_obs_coord[0], next_obs_coord[1]]
        if next_pos_state == self.DEST or next_pos_state == self.DEAD:
            self.env_count[next_obs_coord[0], next_obs_coord[1]] += 1

        if next_pos_state == self.DEAD:
            reward = -100
        # elif next_pos_state == self.DEST:
        #     reward = 100
        else:
            reward = -1

        transition = dict(
            cur_observation=observation,
            cur_action=action,
            next_observation=_coord2obs_index(next_obs_coord, self.env_shape),
            reward=reward,
            terminated=(next_pos_state == self.DEST or next_pos_state == self.DEAD),
        )

        return transition


def _obs_index2_coord(obs_index: int, env_shape: List[int]) -> array:
    width, depth = env_shape

    coords = np.array([obs_index // depth, obs_index % depth], dtype=np.int64)

    return coords


def _coord2obs_index(coord: array, env_shape: List[int]) -> int:
    width, depth = env_shape

    obs_index = coord[0] * depth + coord[1]

    return obs_index


def _action_index2movement(action_index: int, env_dim: int) -> array:
    movement = np.zeros((env_dim,), dtype=np.int64)
    movement[action_index // 2] = -1 if action_index % 2 == 0 else 1

    return movement
