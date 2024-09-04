import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import torch
from labsurv.builders import AGENTS, STRATEGIES
from labsurv.models.agents import BaseAgent
from numpy import ndarray as array
from torch import Tensor
from torch.nn import Module


@AGENTS.register_module()
class OCPREINFORCE(BaseAgent):
    INT = torch.int64
    FLOAT = torch.float

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

        self.action_num = policy_net_cfg.action_dim

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

    def take_action(self, observation: array) -> Tuple[array, array]:
        if self.test_mode:
            return self.test_take_action(observation)
        else:
            return self.train_take_action(observation)

    def train_take_action(self, observation: array) -> Tuple[array, array]:
        observation = torch.tensor(observation, dtype=self.FLOAT, device=self.device)
        with torch.no_grad():  # if grad, memory leaks
            action, params = self.policy_net(observation.unsqueeze(0))

        action_distribution = torch.distributions.Categorical(probs=action)
        chosen_action: array = action_distribution.sample().cpu().numpy().copy()

        # normalization on params: set mu to target interval
        # consider 3sigma as half of the target interval
        params[0] = (params[0] + 1) / 2  # (0, 1)
        params[1] = params[1].clamp(
            min=torch.tensor([0], dtype=self.FLOAT, device=self.device),
            max=torch.tensor([1 / 3], dtype=self.FLOAT, device=self.device),
        )
        pos_dist = torch.distributions.Normal(loc=params[0], scale=params[1])
        params[2] = params[2] * torch.pi  # (-pi, pi)
        params[3] = params[3].clamp(
            min=torch.tensor([0], dtype=self.FLOAT, device=self.device),
            max=torch.tensor([torch.pi / 3], dtype=self.FLOAT, device=self.device),
        )
        pan_dist = torch.distributions.Normal(loc=params[2], scale=params[3])
        params[4] = params[4] * torch.pi / 2  # (-pi/2, pi/2)
        params[5] = params[5].clamp(
            min=torch.tensor([0], dtype=self.FLOAT, device=self.device),
            max=torch.tensor([torch.pi / 6], dtype=self.FLOAT, device=self.device),
        )
        tilt_dist = torch.distributions.Normal(loc=params[4], scale=params[5])
        params[6] = (params[6] + 1) / 2  # (0, 1)
        params[7] = params[7].clamp(
            min=torch.tensor([0], dtype=self.FLOAT, device=self.device),
            max=torch.tensor([1 / 3], dtype=self.FLOAT, device=self.device),
        )
        cam_type_dist = torch.distributions.Normal(loc=params[6], scale=params[7])

        chosen_params: array = (
            torch.cat(
                (  # .sample() returns 0 dimensional tensor (float number-like)
                    pos_dist.sample().unsqueeze(0),
                    pan_dist.sample().unsqueeze(0),
                    tilt_dist.sample().unsqueeze(0),
                    cam_type_dist.sample().unsqueeze(0),
                )
            )
            .cpu()
            .numpy()
            .copy()
        )  # [4]

        return chosen_action, chosen_params

    def test_take_action(self, observation: array) -> Tuple[array, array]:
        observation: Tensor = torch.tensor(
            observation, dtype=self.FLOAT, device=self.device
        )
        with torch.no_grad():
            action, params = self.policy_net(observation.unsqueeze(0))

        return (
            action.argmax().cpu().numpy().copy(),
            params[[0, 2, 4, 6]].cpu().numpy().copy(),
        )

    def update(
        self, markov_chain: Dict[str, List[array | float | Tuple[array, array]]]
    ) -> float:
        cur_observations: List[array] = markov_chain["cur_observation"]
        cur_action_with_params: List[Tuple[array, array]] = markov_chain["cur_action"]
        rewards: List[float] = markov_chain["reward"]

        discounted_reward: float = 0.0
        self.optimizer.zero_grad()
        for step in reversed(range(len(cur_observations))):
            cur_observation: Tensor = torch.tensor(
                cur_observations[step], dtype=self.FLOAT, device=self.device
            )
            cur_action, cur_params = cur_action_with_params[step]
            cur_action: Tensor = torch.tensor(
                cur_action, dtype=self.INT, device=self.device
            )
            cur_params: Tensor = torch.tensor(
                cur_params, dtype=self.FLOAT, device=self.device
            )
            reward: float = rewards[step]

            discounted_reward = self.gamma * discounted_reward + reward
            action_probs, predict_params = self.policy_net(cur_observation.unsqueeze(0))

            loss: Tensor = -discounted_reward * (
                # action
                torch.log(action_probs[cur_action])
                # pos
                - torch.log(torch.tensor([2 * torch.pi], device=self.device)) / 2
                - torch.log(predict_params[1])
                - ((cur_params[0] - predict_params[0]) ** 2)
                / (2 * predict_params[1] ** 2)
                # pan
                - torch.log(torch.tensor([2 * torch.pi], device=self.device)) / 2
                - torch.log(predict_params[3])
                - ((cur_params[1] - predict_params[2]) ** 2)
                / (2 * predict_params[3] ** 2)
                # tilt
                - torch.log(torch.tensor([2 * torch.pi], device=self.device)) / 2
                - torch.log(predict_params[5])
                - ((cur_params[2] - predict_params[4]) ** 2)
                / (2 * predict_params[5] ** 2)
                # cam_type
                - torch.log(torch.tensor([2 * torch.pi], device=self.device)) / 2
                - torch.log(predict_params[7])
                - ((cur_params[3] - predict_params[6]) ** 2)
                / (2 * predict_params[7] ** 2)
            )
            loss.backward()

        self.optimizer.step()

        return loss.item()

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
