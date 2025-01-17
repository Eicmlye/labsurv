import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from labsurv.builders import AGENTS, EXPLORERS, STRATEGIES
from labsurv.models.agents import BaseAgent
from labsurv.models.explorers import BaseExplorer
from numpy import ndarray as array
from torch import Tensor
from torch.nn import Module


@AGENTS.register_module()
class OCPDDPGAddOnlyClean(BaseAgent):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        actor_cfg: Dict,
        critic_cfg: Dict,
        explorer_cfg: Dict,
        device: Optional[str] = None,
        gamma: float = 0.9,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-2,
        tau: float = 5e-2,
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

        if test_mode and load_from is None:
            raise ValueError("`load_from` should not be None in test mode.")
        if test_mode and resume_from is not None:
            raise ValueError("Use `load_from` to load model in test mode.")
        if load_from is not None and resume_from is not None:
            raise ValueError(
                "`load_from` and `resume_from` should not be both specified."
            )

        super().__init__(device, gamma)

        self.test_mode = test_mode
        self.actor_target: Module = STRATEGIES.build(actor_cfg).to(self.device)
        self.critic_target: Module = STRATEGIES.build(critic_cfg).to(self.device)

        if not self.test_mode:
            self.explorer: BaseExplorer = EXPLORERS.build(explorer_cfg)
            self.actor: Module = STRATEGIES.build(actor_cfg).to(self.device)
            self.critic: Module = STRATEGIES.build(critic_cfg).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

            self.lr = [actor_lr, critic_lr]
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr[0])
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr[1])

            self.start_episode = 0
            self.tau = tau

        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)

        self.action_num = actor_cfg.action_dim

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.actor_target.load_state_dict(checkpoint["actor"]["model_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic"]["model_state_dict"])
        if not self.test_mode:
            self.actor.load_state_dict(checkpoint["actor"]["model_state_dict"])
            self.critic.load_state_dict(checkpoint["critic"]["model_state_dict"])
            # One shall not load params of the optimizers, because learning rate
            # is contained in the state_dict of the optimizers, and loading
            # optimizer params will ignore the new learning rate.

    def resume(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.actor_target.load_state_dict(checkpoint["actor"]["model_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic"]["model_state_dict"])
        self.actor.load_state_dict(checkpoint["actor"]["model_state_dict"])
        self.critic.load_state_dict(checkpoint["critic"]["model_state_dict"])
        self.actor_opt.load_state_dict(checkpoint["actor"]["optimizer_state_dict"])
        self.critic_opt.load_state_dict(checkpoint["critic"]["optimizer_state_dict"])
        self.start_episode = checkpoint["episode"] + 1

    def take_action(self, observation: array, all_explore: bool = False) -> array:
        """
        ## Arguments:

            observation (np.ndarray): W * D * H.

        ## Returns:

            action_with_params (np.ndarray): [7], when take action, batch is always 1
        """

        if self.test_mode:
            return self.test_take_action(observation)
        else:
            if all_explore or self.explorer.decide(observation):
                # [7], when take action, batch is always 1
                return (
                    _pos_index2coord(
                        torch.tensor(observation[0], device=self.device),
                        torch.tensor(
                            self.explorer.act(observation), device=self.device
                        ),
                    )
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .copy()
                )
            else:
                return self.train_take_action(observation)

    def train_take_action(self, observation: array) -> array:
        """
        ## Arguments:

            observation (np.ndarray): W * D * H.

        ## Returns:

            action_with_params (np.ndarray): [7], when take action, batch is always 1
        """

        observation: Tensor = torch.tensor(
            observation, dtype=self.FLOAT, device=self.device
        )
        with torch.no_grad():  # if grad, memory leaks
            x, pos_mask = _observation2input(observation.unsqueeze(0))
            action_with_params: Tensor = self.actor(x, pos_mask)  # B * 7

        # [7], when take action, batch is always 1
        return action_with_params.squeeze(0).cpu().numpy().copy()

    def test_take_action(self, observation: array) -> array:
        """
        ## Arguments:

            observation (np.ndarray): W * D * H.

        ## Returns:

            action_with_params (np.ndarray): [7], when take action, batch is always 1
        """

        observation: Tensor = torch.tensor(
            observation, dtype=self.FLOAT, device=self.device
        )
        with torch.no_grad():
            x, pos_mask = _observation2input(observation.unsqueeze(0))
            action_with_params: Tensor = self.actor(x, pos_mask)  # B * 7

        # [7], when take action, batch is always 1
        return action_with_params.squeeze(0).cpu().numpy().copy()

    def update(
        self, samples: Dict[str, List[bool | float | array | Tuple[int, array]]]
    ) -> List[float]:
        cur_observations: List[array] = samples["cur_observation"]
        cur_action_with_params: List[Tuple[int, array]] = samples["cur_action"]
        rewards: List[float] = samples["reward"]
        next_observations: List[array] = samples["next_observation"]
        terminated: List[bool] = samples["terminated"]

        cur_observations: Tensor = torch.tensor(
            np.array(cur_observations), dtype=self.FLOAT, device=self.device
        )
        cur_action_with_params: Tensor = torch.tensor(
            np.array(cur_action_with_params), dtype=self.FLOAT, device=self.device
        )
        rewards: Tensor = torch.tensor(
            np.array(rewards), dtype=self.FLOAT, device=self.device
        )
        next_observations: Tensor = torch.tensor(
            np.array(next_observations), dtype=self.FLOAT, device=self.device
        )
        terminated: Tensor = torch.tensor(
            np.array(terminated), dtype=self.INT, device=self.device
        )

        # critic weights update
        next_x, next_pos_mask = _observation2input(next_observations)
        target_q = self.critic_target(
            next_x.clone().detach(),
            self.actor_target(next_x.clone().detach(), next_pos_mask),
        ).squeeze(
            1
        )  # [B]
        discounted_target_q = rewards + self.gamma * target_q * (1 - terminated)
        x, pos_mask = _observation2input(cur_observations)
        critic_loss = torch.mean(
            F.mse_loss(
                self.critic(x.clone().detach(), cur_action_with_params).squeeze(
                    1
                ),  # [B]
                discounted_target_q,
            )
        )
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # actor weights update
        actor_loss = -torch.mean(
            self.critic(x.clone().detach(), self.actor(x.clone().detach(), pos_mask))
        )
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # soft update target nets
        for param_target, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )
        for param_target, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )

        return critic_loss.item(), actor_loss.item()

    def save(self, episode_index: int, save_path: str):
        checkpoint = dict(
            actor=dict(
                model_state_dict=self.actor_target.state_dict(),
                optimizer_state_dict=self.actor_opt.state_dict(),
            ),
            critic=dict(
                model_state_dict=self.critic_target.state_dict(),
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


def _pos_index2coord(occupancy: Tensor, action_with_params: Tensor):
    """
    ## Description:

        Change pos_index in params to pos_coord, making 5-elem params to 7-elem.
    """
    assert action_with_params.ndim == 2 and action_with_params.shape[1] == 5, (
        "`action` should be shaped [B, 5], " f"but got {action_with_params.shape}."
    )
    pos = (occupancy + 1).nonzero()[action_with_params[:, 1].type(torch.int64)]
    action_with_params = torch.cat(
        (
            action_with_params[:, [0]],
            pos,
            action_with_params[:, [2, 3, 4]],
        ),
        dim=1,
    )

    return action_with_params  # B * 7


def _observation2input(observation: Tensor) -> Tuple[Tensor, Tensor]:
    x: Tensor = observation.clone().detach()

    # 1 for blocked, 2 for visible, 0 for invisible
    x = (x[:, 0] + x[:, -1] * 2).unsqueeze(1)  # [B, 1, W, H ,D]

    cache_observ: Tensor = observation.clone().detach()
    # pos that allows installation and yet haven't been installed at
    pos_mask: Tensor = torch.logical_xor(cache_observ[:, [1]], cache_observ[:, [7]])

    return x, pos_mask
