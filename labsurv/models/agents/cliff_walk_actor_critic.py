import os
import os.path as osp
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from labsurv.builders import AGENTS
from labsurv.models.agents import BaseAgent
from labsurv.runners.hooks import LoggerHook
from numpy import ndarray as array
from torch import Tensor
from torch.nn import Linear, Module


class Actor(Module):
    def __init__(self, device, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.device = torch.device(device)

        self.linear = Linear(state_dim, hidden_dim, device=self.device)
        self.mid = Linear(hidden_dim, hidden_dim, device=self.device)
        self.out = Linear(hidden_dim, action_dim, device=self.device)

    def forward(self, input: Tensor):
        obs_indices = input.type(torch.int64)
        x = (
            F.one_hot(obs_indices, num_classes=self.linear.in_features)
            .type(torch.float)
            .view(-1, self.linear.in_features)
        )

        x = F.relu(self.linear(x))
        x = F.relu(self.mid(x))

        out = F.softmax(self.out(x), dim=1)

        return out


class Critic(Module):
    def __init__(self, device, state_dim, hidden_dim):
        super().__init__()
        self.device = torch.device(device)

        self.linear = Linear(state_dim, hidden_dim, device=self.device)
        self.mid = Linear(hidden_dim, hidden_dim, device=self.device)
        self.out = Linear(hidden_dim, 1, device=self.device)

    def forward(self, input: Tensor):
        obs_indices = input.type(torch.int64)
        x = (
            F.one_hot(obs_indices, num_classes=self.linear.in_features)
            .type(torch.float)
            .view(-1, self.linear.in_features)
        )

        x = F.relu(self.linear(x))
        x = F.relu(self.mid(x))

        out = self.out(x)

        return out


@AGENTS.register_module()
class CliffWalkActorCritic(BaseAgent):
    FREE = 0
    FROM = 1
    DEAD = 2
    DEST = 3

    def __init__(
        self,
        device: str,
        gamma: float,
        actor_cfg: Dict,
        critic_cfg: Optional[Dict] = None,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        entropy_loss_coef: float = 0.01,
        advantage_coef: float = 0.95,
        load_from: str = None,
        resume_from: str = None,
        test_mode: bool = False,
    ):
        super().__init__(device, gamma)

        self.test_mode = test_mode
        self.actor: Module = Actor(**actor_cfg).to(self.device)

        if not self.test_mode:
            self.entropy_loss_coef = entropy_loss_coef
            self.advantage_coef = advantage_coef

            self.critic: Module = Critic(**critic_cfg).to(self.device)

            self.lr = [actor_lr, critic_lr]
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr[0])
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr[1])

            self.start_episode = 0

        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)

    def train(self):
        self.actor.train()
        self.critic.train()
        self.test_mode = False

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.test_mode = True

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        # load model weights
        if not self.test_mode:
            self.critic.load_state_dict(checkpoint["critic"])
        self.actor.load_state_dict(checkpoint["actor"])

    def resume(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        # load model weights
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt"])

        self.start_episode = checkpoint["episode"] + 1

    def take_action(self, observation, **kwargs):
        if self.test_mode:
            return self.test_take_action(observation, **kwargs)
        else:
            return self.train_take_action(observation, **kwargs)

    def train_take_action(self, observation, **kwargs):
        if "logger" in kwargs.keys():
            logger: LoggerHook = kwargs["logger"]

        with torch.no_grad():
            obs: Tensor = torch.tensor(
                observation, dtype=torch.float, device=self.device
            ).view(-1, 1)

            action_probs: Tensor = self.actor(obs).squeeze(0)
            action_dist = torch.distributions.Categorical(action_probs)
            if "episode_index" in kwargs.keys() and "step_index" in kwargs.keys():
                episode = kwargs["episode_index"] + 1
                step = kwargs["step_index"] + 1
                logger.show_log(
                    f"[Episode {episode:>4}  Step {step:>3}]"
                    "\n    ^         v         <         >    "
                    f"\n{action_probs[0] * 100:2.4f}%  "
                    f"{action_probs[1] * 100:2.4f}%  "
                    f"{action_probs[2] * 100:2.4f}%  "
                    f"{action_probs[3] * 100:2.4f}%  ",
                    with_time=True,
                )

            action_index: int = action_dist.sample().type(torch.int64).item()

        return action_index

    def test_take_action(self, observation, **kwargs):
        with torch.no_grad():
            obs: Tensor = torch.tensor(
                observation, dtype=torch.float, device=self.device
            ).view(-1, 1)

            action_probs: Tensor = self.actor(obs).squeeze(0)

            action_index: int = action_probs.argmax().type(torch.int64).item()

        return action_index

    def update(self, transitions: dict, logger: LoggerHook):
        cur_observations: Tensor = torch.tensor(
            transitions["cur_observation"], dtype=torch.float, device=self.device
        ).view(-1, 1)
        cur_actions: Tensor = torch.tensor(
            transitions["cur_action"], dtype=torch.int64, device=self.device
        ).view(-1, 1)
        next_observations: Tensor = torch.tensor(
            transitions["next_observation"], dtype=torch.float, device=self.device
        ).view(-1, 1)
        rewards: Tensor = torch.tensor(
            transitions["reward"], dtype=torch.float, device=self.device
        ).view(-1, 1)
        terminated: Tensor = torch.tensor(
            transitions["terminated"], dtype=torch.float, device=self.device
        ).view(-1, 1)

        td_target: Tensor = rewards + self.gamma * self.critic(next_observations) * (
            1 - terminated
        )
        td_error: Tensor = td_target - self.critic(cur_observations)
        # advantages: Tensor = _compute_advantage(
        #     self.gamma, self.advantage_coef, td_error, self.device
        # )

        log_probs: Tensor = torch.log(
            torch.gather(
                self.actor(cur_observations),
                dim=1,
                index=cur_actions,
            )
        )

        actor_loss: Tensor = torch.mean(-log_probs * td_error.detach())
        critic_loss: Tensor = torch.mean(
            F.mse_loss(self.critic(cur_observations), td_target.detach())
        )
        entropy_loss: Tensor = torch.mean(
            torch.distributions.Categorical(self.actor(cur_observations)).entropy()
        )

        loss: Tensor = (
            actor_loss + 0.5 * critic_loss - self.entropy_loss_coef * entropy_loss
        )

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()

        return actor_loss.item(), critic_loss.item(), entropy_loss.item()

    def save(self, episode_index: int, save_path: str):
        checkpoint = dict(
            actor=self.actor.state_dict(),
            actor_opt=self.actor_opt.state_dict(),
            critic=self.critic.state_dict(),
            critic_opt=self.critic_opt.state_dict(),
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

    def print_strat(self, env: array, logger: LoggerHook):
        logger.show_log("========")
        width, depth = env.shape

        obs_indices = torch.tensor(
            [i for i in range(width * depth)], dtype=torch.float, device=self.device
        ).view(-1, 1)

        actions = (
            torch.argmax(
                self.actor(obs_indices),
                dim=1,
            )
            .view(-1)
            .tolist()
        )

        for w in range(width):
            for d in range(depth):
                if env[w, d] == self.DEAD:
                    logger.show_log("x", end="  ")
                else:
                    index = w * depth + d

                    if actions[index] == 0:
                        logger.show_log("^", end="  ")
                    elif actions[index] == 1:
                        logger.show_log("v", end="  ")
                    elif actions[index] == 2:
                        logger.show_log("<", end="  ")
                    elif actions[index] == 3:
                        logger.show_log(">", end="  ")

            logger.show_log("")

        logger.show_log("========")


def _compute_advantage(
    gamma: float, advantage_coef: float, td_error: Tensor, device: torch.cuda.device
):
    td_error = td_error.clone().detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0

    for delta in td_error[::-1]:
        advantage = gamma * advantage_coef * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()

    return torch.tensor(
        np.array(advantage_list), dtype=torch.float, device=device
    )  # [B]
