import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from labsurv.builders import AGENTS, STRATEGIES
from labsurv.models.agents import BaseAgent
from labsurv.runners.hooks import LoggerHook
from labsurv.utils.surveillance import action_index2movement
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
        entropy_loss_coef: float = 0.01,
        cam_types: int = 1,
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
        self.cam_types = cam_types

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
            self.entropy_loss_coef = entropy_loss_coef

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

            observation (np.ndarray): [1, 6].

        ## Returns:

            action (np.ndarray): [7].
        """

        if self.test_mode:
            return self.test_take_action(observation, **kwargs)
        else:
            return self.train_take_action(observation, **kwargs)

    def train_take_action(self, observation: array, **kwargs) -> array:
        """
        ## Arguments:

            observation (np.ndarray): [1, 6].

        ## Returns:

            action (np.ndarray): [7].
        """

        with torch.no_grad():  # if grad, memory leaks
            obs: Tensor = torch.tensor(  # [1, 6]
                observation, dtype=self.FLOAT, device=self.device
            )

            action_dist: Tensor = self.actor(obs, without_noise=self.test_mode)
            action_dist_cat = torch.distributions.Categorical(action_dist)

            # DEBUG(eric)
            print(
                f"[Episode {kwargs["episode_index"] + 1:>4}  "
                f"Step {kwargs["step_index"] + 1:>3}] action distribution"
                "\n           x         y         z         p         t",
                end="",
            )
            action_probs = [float(i) for i in action_dist[0].tolist()]
            print("\n   -   ", end="")
            for i in range(5):
                print(f"{action_probs[2 * i]:.6f}  ", end="")
            print("\n   +   ", end="")
            for i in range(5):
                print(f"{action_probs[2 * i + 1]:.6f}  ", end="")
            print(f"\nEntropy: {action_dist_cat.entropy().item():.6f}")

            action_index: int = action_dist_cat.sample().type(torch.int64).item()

            movement: array = action_index2movement(action_index, self.cam_types)

        return np.concatenate((movement, np.array([action_index])), axis=0)

    def test_take_action(self, observation: array, **kwargs) -> array:
        """
        ## Arguments:

            observation (np.ndarray): [1, 6].

        ## Returns:

            action (np.ndarray): [7].
        """

        with torch.no_grad():  # if grad, memory leaks
            obs: Tensor = torch.tensor(  # [1, 6]
                observation, dtype=self.FLOAT, device=self.device
            )
            action_dist: Tensor = self.actor(obs, without_noise=self.test_mode)

            # DEBUG(eric)
            print(
                f"[Step {kwargs["step_index"] + 1:>3}] action distribution"
                "\n           x         y         z         p         t",
                end="",
            )
            action_dist_cat = torch.distributions.Categorical(action_dist)
            action_probs = [float(i) for i in action_dist[0].tolist()]
            print("\n   -   ", end="")
            for i in range(5):
                print(f"{action_probs[2 * i]:.6f}  ", end="")
            print("\n   +   ", end="")
            for i in range(5):
                print(f"{action_probs[2 * i + 1]:.6f}  ", end="")
            print(f"\nEntropy: {action_dist_cat.entropy().item():.6f}")

            action_index: int = action_dist.argmax().type(torch.int64).item()

            movement: array = action_index2movement(action_index, self.cam_types)

        return np.concatenate((movement, np.array([action_index])), axis=0)

    def update(
        self,
        transitions: Dict[str, List[bool | float | array | Tuple[int, array]]],
        logger: LoggerHook,
    ) -> Tuple[float]:
        cur_observations: List[array] = transitions["cur_observation"]
        cur_action: List[array] = transitions["cur_action"]
        rewards: List[float] = transitions["reward"]
        next_observations: List[array] = transitions["next_observation"]
        terminated: List[bool] = transitions["terminated"]

        cur_observations: Tensor = torch.tensor(
            np.array(cur_observations), dtype=self.FLOAT, device=self.device
        )  # [B, 6]
        cur_action: Tensor = torch.tensor(
            np.array(cur_action), dtype=self.FLOAT, device=self.device
        )  # [B, 7]
        rewards: Tensor = torch.tensor(
            np.array(rewards), dtype=self.FLOAT, device=self.device
        )  # [B]
        next_observations: Tensor = torch.tensor(
            np.array(next_observations), dtype=self.FLOAT, device=self.device
        )  # [B, 6]
        terminated: Tensor = torch.tensor(
            np.array(terminated), dtype=self.INT, device=self.device
        )  # [B]
        cur_action_indices: Tensor = (
            cur_action[:, 6].view(-1, 1).type(self.INT)
        )  # [B, 1]

        # critic weights update
        value_predict: Tensor = self.critic(next_observations).squeeze(1)  # [B]
        td_target = rewards + self.gamma * value_predict * (1 - terminated)  # [B]
        clipped_td_target = torch.clamp(td_target, -500, 500)
        td_error = clipped_td_target - self.critic(cur_observations)  # [B]
        advantages = _compute_advantage(  # [B]
            self.gamma,
            self.advantage_param,
            td_error,
            self.device,
        )

        pred_strat_prob = torch.log(
            torch.gather(
                self.actor(cur_observations, without_noise=True),
                dim=1,
                index=cur_action_indices,
            ).squeeze(1)
        ).detach()  # [B]

        for step in range(self.update_step):
            action_dist: Tensor = self.actor(cur_observations, without_noise=True)
            cur_strat_prob = torch.log(
                torch.gather(
                    action_dist,
                    dim=1,
                    index=cur_action_indices,
                ).squeeze(1)
            )  # [B]
            entropy: Tensor = torch.distributions.Categorical(action_dist).entropy()

            significance = torch.exp(cur_strat_prob - pred_strat_prob)  # [B]

            surrogate_1 = significance * advantages
            surrogate_2 = (
                torch.clamp(
                    significance,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon,
                )
                * advantages
            )  # PPO clip

            actor_loss = torch.mean(-torch.min(surrogate_1, surrogate_2))  # PPO loss
            critic_loss = torch.mean(
                F.mse_loss(
                    self.critic(cur_observations).squeeze(1),
                    clipped_td_target.detach(),
                )
            )
            entropy_loss = entropy.mean()
            total_loss = (
                actor_loss + 0.5 * critic_loss - self.entropy_loss_coef * entropy_loss
            )

            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            total_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()

            logger.show_log(
                f"[Update step {step + 1:>3}]  loss: C {critic_loss.item():.6f} "
                f"A {actor_loss.item():.6f} E {entropy_loss.item():.6f}",
                with_time=True,
            )

        return critic_loss.item(), actor_loss.item(), entropy_loss.item()

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
