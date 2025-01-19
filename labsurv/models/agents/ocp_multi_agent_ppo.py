import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from labsurv.builders import AGENTS, STRATEGIES
from labsurv.models.agents import BaseAgent
from labsurv.runners.hooks import LoggerHook
from labsurv.utils.surveillance import (
    direction_index2pan_tilt,
    pos_index2coord,
    visualize_distribution_heatmap,
)
from numpy import ndarray as array
from torch import Tensor
from torch.nn import Module


@AGENTS.register_module()
class OCPMultiAgentPPO(BaseAgent):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        actor_cfg: Dict,
        critic_cfg: Dict,
        device: Optional[str] = None,
        gamma: float = 0.9,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-4,
        update_step: int = 10,
        advantage_param: float = 0.95,
        clip_epsilon: float = 0.2,
        pan_section_num: int = 360,
        tilt_section_num: int = 180,
        load_from: Optional[str] = None,
        resume_from: Optional[str] = None,
        test_mode: bool = False,
    ):
        """
        The following combinations to specify arguments are allowed:
        1. `load_from`: train the agent from `load_from` with a new optimizer.
        2. `resume_from`: train the agent from `resume_from` with the exact optimizer
            `resume_from` was using.
        3. `load_from`, `test_mode`: test the agent from `load_from`.
        4. None of the above specified: train a brand new agent with a new optimizer.
        """

        if load_from is not None and resume_from is not None:
            raise ValueError(
                "`load_from` and `resume_from` should not be both specified."
            )
        if test_mode and load_from is None:
            raise ValueError("`load_from` should not be None in test mode.")
        if test_mode and resume_from is not None:
            raise ValueError(
                "Use `load_from` instead of `resume_from` to load model in test mode."
            )

        super().__init__(device, gamma)

        self.test_mode = test_mode
        self.pan_section_num = pan_section_num
        self.tilt_section_num = tilt_section_num

        self.actor: Module = STRATEGIES.build(actor_cfg).to(self.device)
        self.critic: Module = STRATEGIES.build(critic_cfg).to(self.device)

        if not self.test_mode:
            self.lr = [actor_lr, critic_lr]
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr[0])
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr[1])

            self.start_episode = 0
            self.update_step = update_step
            self.advantage_param = advantage_param
            self.clip_epsilon = clip_epsilon

        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)

    def eval(self):
        self.test_mode = True
        self.actor.eval()

    def train(self):
        self.test_mode = False
        self.actor.train()

    @property
    def direction_num(self):
        return self.pan_section_num * (self.tilt_section_num - 1) + 1

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.actor.load_state_dict(checkpoint["actor"]["model_state_dict"])
        self.critic.load_state_dict(checkpoint["critic"]["model_state_dict"])
        # One shall not load params of the optimizers, because learning rate
        # is contained in the state_dict of the optimizers, and loading
        # optimizer params will ignore the new learning rate.

    def resume(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.actor.load_state_dict(checkpoint["actor"]["model_state_dict"])
        self.critic.load_state_dict(checkpoint["critic"]["model_state_dict"])
        self.actor_opt.load_state_dict(checkpoint["actor"]["optimizer_state_dict"])
        self.critic_opt.load_state_dict(checkpoint["critic"]["optimizer_state_dict"])
        self.start_episode = checkpoint["episode"] + 1

    def take_action(self, observation: array, **kwargs) -> array:
        """
        ## Arguments:

            observation (np.ndarray): [1, 2, W, D, H].

        ## Returns:

            action_with_params (np.ndarray): [9].
        """

        if self.test_mode:
            return self.test_take_action(observation, **kwargs)
        else:
            return self.train_take_action(observation, **kwargs)

    def train_take_action(self, observation: array, **kwargs) -> array:
        """
        ## Arguments:

            observation (np.ndarray): [1, 2, W, D, H].

        ## Returns:

            action_with_params (np.ndarray): [9].
        """

        room_shape = observation.shape[2:]
        with torch.no_grad():  # if grad, memory leaks
            obs: Tensor = torch.tensor(  # [1, 2, W, D, H]
                observation, dtype=self.FLOAT, device=self.device
            )
            # [1, 1, W, D, H]
            x, pos_mask = obs[:, 0].unsqueeze(1), obs[:, 1].unsqueeze(1)
            pos_dist, param_dist = self.actor(x, pos_mask)

            if "step_index" in kwargs.keys():
                width, depth, height = x.shape[2:]

                visualize_distribution_heatmap(
                    pos_dist.squeeze(0).view(width, depth, height),
                    osp.join(
                        kwargs["save_dir"],
                        "dist",
                        "train",
                    ),
                    f"epi{kwargs["episode_index"] + 1}_step{kwargs["step_index"] + 1}",
                )

            pos_index = (
                torch.distributions.Categorical(pos_dist[0])
                .sample()
                .type(self.INT)
                .item()
            )
            pos_coord = pos_index2coord(room_shape, pos_index, self.device).tolist()

            param_index = (
                torch.distributions.Categorical(param_dist[0])
                .sample()
                .type(self.INT)
                .item()
            )
            direction_index = param_index % self.direction_num
            direction = direction_index2pan_tilt(
                self.pan_section_num,
                self.tilt_section_num,
                direction_index,
                self.device,
            ).tolist()

            cam_type = param_index // self.direction_num

        return np.array(
            [0] + pos_coord + direction + [cam_type, pos_index, direction_index],
            dtype=np.float32,
        )

    def test_take_action(self, observation: array, **kwargs) -> array:
        """
        ## Arguments:

            observation (np.ndarray): [1, 2, W, D, H].

        ## Returns:

            action_with_params (np.ndarray): [9].
        """

        room_shape = observation.shape[2:]
        with torch.no_grad():  # if grad, memory leaks
            obs: Tensor = torch.tensor(  # [1, 2, W, D, H]
                observation, dtype=self.FLOAT, device=self.device
            )
            # [1, 1, W, D, H]
            x, pos_mask = obs[:, 0].unsqueeze(1), obs[:, 1].unsqueeze(1)
            pos_dist, param_dist = self.actor(x, pos_mask)

            if "step_index" in kwargs.keys():
                width, depth, height = x.shape[2:]
                train_mode = (
                    "episode_index" in kwargs.keys()
                    and kwargs["episode_index"] is not None
                )

                visualize_distribution_heatmap(
                    pos_dist.squeeze(0).view(width, depth, height),
                    osp.join(
                        kwargs["save_dir"],
                        "dist",
                        "eval" if train_mode else "test",
                    ),
                    (f"epi{kwargs["episode_index"] + 1}_" if train_mode else "")
                    + f"step{kwargs["step_index"] + 1}",
                )

            pos_index = pos_dist[0].argmax().type(self.INT).item()
            pos_coord = pos_index2coord(room_shape, pos_index, self.device).tolist()

            param_index = param_dist[0].argmax().type(self.INT).item()
            direction_index = param_index % self.direction_num
            direction = direction_index2pan_tilt(
                self.pan_section_num,
                self.tilt_section_num,
                direction_index,
                self.device,
            ).tolist()

            cam_type = param_index // self.direction_num

        return np.array(
            [0] + pos_coord + direction + [cam_type, pos_index, direction_index],
            dtype=np.float32,
        )

    def update(
        self,
        transitions: Dict[str, List[bool | float | array | Tuple[int, array]]],
        logger: LoggerHook,
    ) -> List[float]:
        cur_observations: List[array] = transitions["cur_observation"]
        cur_action_with_params: List[array] = transitions["cur_action"]
        rewards: List[float] = transitions["reward"]
        next_observations: List[array] = transitions["next_observation"]
        terminated: List[bool] = transitions["terminated"]

        cur_observations: Tensor = torch.tensor(
            np.array(cur_observations), dtype=self.FLOAT, device=self.device
        )  # [B, 2, W, D, H]
        cur_action_with_params: Tensor = torch.tensor(
            np.array(cur_action_with_params), dtype=self.FLOAT, device=self.device
        )  # [B, 9]
        rewards: Tensor = torch.tensor(
            np.array(rewards), dtype=self.FLOAT, device=self.device
        )  # [B]
        next_observations: Tensor = torch.tensor(
            np.array(next_observations), dtype=self.FLOAT, device=self.device
        )  # [B, 2, W, D, H]
        terminated: Tensor = torch.tensor(
            np.array(terminated), dtype=self.INT, device=self.device
        )  # [B]
        cur_action_pos_indices: Tensor = (
            cur_action_with_params[:, 7].view(-1, 1).type(self.INT)
        )  # [B, 1]
        cur_action_param_indices: Tensor = (
            cur_action_with_params[:, 8].view(-1, 1).type(self.INT)
        )  # [B, 1]

        # critic weights update
        next_x = next_observations[:, 0].unsqueeze(1)  # [B, 1, W, D, H]
        value_predict: Tensor = self.critic(next_x).squeeze(1)  # [B]
        td_target = rewards + self.gamma * value_predict * (1 - terminated)  # [B]
        # [B, 1, W, D, H]
        x, pos_mask = cur_observations[:, 0].unsqueeze(1), cur_observations[
            :, 1
        ].unsqueeze(1)
        td_error = td_target - self.critic(x)  # [B]
        advantage = _compute_advantage(  # [B]
            self.gamma,
            self.advantage_param,
            td_error,
            self.device,
        )

        # [B, W * D * H], [B, DIRECTION * CAM_TYPE]
        pred_pos_dist, pred_param_dist = self.actor(x, pos_mask)
        pred_strat_prob = (
            (
                torch.gather(pred_pos_dist, dim=1, index=cur_action_pos_indices)
                * torch.gather(pred_param_dist, dim=1, index=cur_action_param_indices)
            )
            .squeeze(1)
            .detach()
        )  # [B]

        for step in range(self.update_step):
            # [B, W * D * H], [B, DIRECTION * CAM_TYPE]
            cur_pos_dist, cur_param_dist = self.actor(x, pos_mask)
            cur_strat_prob = (
                torch.gather(cur_pos_dist, dim=1, index=cur_action_pos_indices)
                * torch.gather(cur_param_dist, dim=1, index=cur_action_param_indices)
            ).squeeze(
                1
            )  # [B]

            significance = cur_strat_prob / pred_strat_prob  # [B]

            trust_bound_1 = significance * advantage
            trust_bound_2 = (
                torch.clamp(
                    significance,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon,
                )
                * advantage
            )  # PPO clip

            actor_loss = torch.mean(
                -torch.min(trust_bound_1, trust_bound_2)
            )  # PPO loss
            critic_loss = torch.mean(
                F.mse_loss(self.critic(x).squeeze(1), td_target.detach())
            )
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()

            logger.show_log(
                f"[Update step {step + 1:>3}]  loss: C {critic_loss.item():.4f} "
                f"A {actor_loss.item():.4f}",
                with_time=True,
            )

        return critic_loss.item(), actor_loss.item()

    def save(self, episode_index: int, save_path: str):
        checkpoint = dict(
            actor=dict(
                model_state_dict=self.actor.state_dict(),
                optimizer_state_dict=self.actor_opt.state_dict(),
            ),
            critic=dict(
                model_state_dict=self.critic.state_dict(),
                optimizer_state_dict=self.critic_opt.state_dict(),
            ),
            episode=episode_index,
        )

        episode = episode_index + 1
        if save_path.endswith(".pth"):
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            save_path = ".".join(save_path.split(".")[:-1]) + f"_episode_{episode}.pth"
        else:
            os.makedirs(save_path, exist_ok=True)
            save_path = osp.join(save_path, f"episode_{episode}.pth")

        torch.save(checkpoint, save_path)


def _compute_advantage(
    gamma: float, advantage_param: float, td_error: Tensor, device: torch.cuda.device
):
    td_error = td_error.clone().detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0

    for delta in td_error[::-1]:
        advantage = gamma * advantage_param * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()

    return torch.tensor(
        np.array(advantage_list), dtype=torch.float, device=device
    )  # [B]
