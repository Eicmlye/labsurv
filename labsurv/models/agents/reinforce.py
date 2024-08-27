import os
import os.path as osp
from typing import Dict, Optional

import numpy as np
import torch
from labsurv.builders import AGENTS, STRATEGIES
from labsurv.models.agents import BaseAgent
from torch import Tensor
from torch.nn import Module


@AGENTS.register_module()
class REINFORCE(BaseAgent):
    def __init__(
        self,
        policy_net_cfg: Dict,
        device: Optional[torch.cuda.device] = None,
        gamma: float = 0.9,
        lr: float = 0.1,
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
        self.policy_net: Module = STRATEGIES.build(policy_net_cfg).to(self.device)

        if not self.test_mode:
            self.lr = lr
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
            self.start_episode = 0

        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        if not self.test_mode:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def resume(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_episode = checkpoint["episode"]

    def take_action(self, observation):
        if self.test_mode:
            return self.test_take_action(observation)
        else:
            return self.train_take_action(observation)

    def train_take_action(self, observation):
        observation = Tensor(observation).to(self.device)
        actions = self.policy_net(observation)
        action_distribution = torch.distributions.Categorical(probs=actions)

        return action_distribution.sample().item()

    def test_take_action(self, observation):
        observation = Tensor(observation).to(self.device)
        action = self.policy_net(observation).argmax().item()

        return action

    def update(self, markov_chain) -> Tensor:
        cur_observations = markov_chain["cur_observation"]
        cur_actions = markov_chain["cur_action"]
        rewards = markov_chain["reward"]

        discounted_reward = Tensor([0]).to(self.device)
        self.optimizer.zero_grad()
        for step in reversed(range(len(cur_observations))):
            cur_observation = Tensor(np.array(cur_observations[step])).to(self.device)
            cur_action = Tensor([cur_actions[step]]).type(torch.int64).to(self.device)
            reward = Tensor([rewards[step]]).to(self.device)

            discounted_reward = self.gamma * discounted_reward + reward
            loss = -discounted_reward * torch.log(
                self.policy_net(cur_observation).gather(0, cur_action)
            )

            loss.backward()
        self.optimizer.step()

        return loss

    def save(self, episode_index: int, save_path: str):
        checkpoint = dict(
            model_state_dict=self.policy_net.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
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
