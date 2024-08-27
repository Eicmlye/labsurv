import os
import os.path as osp
from typing import Dict, Optional

import torch
from labsurv.builders import AGENTS, EXPLORERS, STRATEGIES
from labsurv.models.agents import BaseAgent
from labsurv.models.explorers import BaseExplorer
from torch import Tensor
from torch.nn import Module


@AGENTS.register_module()
class DQN(BaseAgent):
    def __init__(
        self,
        qnet_cfg: Dict,
        device: torch.cuda.device = None,
        gamma: float = 0.9,
        explorer_cfg: Dict = None,
        lr: float = 0.1,
        to_target_net_interval: int = 5,
        dqn_type: str = "DQN",
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

        self.device = device
        self.gamma = gamma

        self.test_mode = test_mode
        self.target_net: Module = STRATEGIES.build(qnet_cfg).to(self.device)

        if not self.test_mode:
            if explorer_cfg is not None:
                explorer_cfg["samples"] = range(qnet_cfg.action_dim)
                self.explorer: BaseExplorer = EXPLORERS.build(explorer_cfg)

            self.qnet: Module = STRATEGIES.build(qnet_cfg).to(self.device)
            self.lr = lr
            self.to_target_net_interval = to_target_net_interval
            self.update_count = 0
            self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
            self.start_episode = 0

        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)

        if dqn_type not in ["DQN", "DoubleDQN"]:
            raise NotImplementedError(f"{dqn_type} not implemented.")
        self.dqn_type = dqn_type

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.target_net.load_state_dict(checkpoint["model_state_dict"])
        if not self.test_mode:
            self.qnet.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def resume(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.target_net.load_state_dict(checkpoint["model_state_dict"])
        self.qnet.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_episode = checkpoint["episode"]

    def take_action(self, observation):
        if self.test_mode:
            return self.test_take_action(observation)
        else:
            return self.train_take_action(observation)

    def train_take_action(self, observation):
        if self.explorer.decide():
            return self.explorer.act()
        else:
            observation = Tensor(observation).to(self.device)
            action = self.qnet(observation).argmax().item()

            return action

    def test_take_action(self, observation):
        observation = Tensor(observation).to(self.device)
        action = self.target_net(observation).argmax().item()

        return action

    def update(self, samples) -> Tensor:
        cur_observations = torch.row_stack(samples["cur_observation"])
        cur_actions = (
            torch.row_stack(samples["cur_action"]).view(-1, 1).type(torch.int64)
        )
        rewards = torch.row_stack(samples["reward"]).view(-1, 1)
        next_observations = torch.row_stack(samples["next_observation"])
        terminated = torch.row_stack(samples["terminated"]).view(-1, 1)

        total_rewards = self.qnet(cur_observations).gather(dim=1, index=cur_actions)
        if self.dqn_type == "DQN":
            max_next_total_rewards = (
                self.target_net(next_observations).max(dim=1)[0].view(-1, 1)
            )
        elif self.dqn_type == "DoubleDQN":
            max_action = (
                self.qnet(next_observations).max(1)[1].view(-1, 1).type(torch.int64)
            )
            max_next_total_rewards = self.target_net(next_observations).gather(
                1, max_action
            )

        q_targets = rewards + self.gamma * max_next_total_rewards * (1 - terminated)

        self.optimizer.zero_grad()
        loss = self.qnet.get_loss(total_rewards, q_targets)
        loss.backward()
        self.optimizer.step()

        if self.update_count % self.to_target_net_interval == 0:
            self.target_net.load_state_dict(self.qnet.state_dict())
        self.update_count += 1

        return loss

    def save(self, episode_index: int, save_path: str):
        checkpoint = dict(
            model_state_dict=self.target_net.state_dict(),
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
