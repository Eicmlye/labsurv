import os
import os.path as osp

import torch
import torch.nn.functional as F
from labsurv.builders import AGENTS, EXPLORERS
from labsurv.models.agents import BaseAgent
from labsurv.models.explorers import BaseExplorer
from labsurv.runners.hooks import LoggerHook
from numpy import ndarray as array
from torch import Tensor
from torch.nn import Linear, Module


class QNet(Module):
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

        out = self.out(x)

        return out


@AGENTS.register_module()
class CliffWalkDQN(BaseAgent):
    FREE = 0
    FROM = 1
    DEAD = 2
    DEST = 3

    def __init__(
        self,
        device,
        gamma,
        qnet_cfg,
        lr,
        explorer_cfg,
        load_from: str = None,
        resume_from: str = None,
        test_mode: bool = False,
    ):
        super().__init__(device, gamma)

        self.test_mode = test_mode
        self.target_qnet: Module = QNet(**qnet_cfg).to(self.device)

        if not self.test_mode:
            self.explorer: BaseExplorer = EXPLORERS.build(explorer_cfg)
            self.qnet: Module = QNet(**qnet_cfg).to(self.device)
            self.target_qnet.load_state_dict(self.qnet.state_dict())

            self.lr = lr
            self.opt = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

            self.start_episode = 0

        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)

    def train(self):
        self.qnet.train()
        self.test_mode = False

    def eval(self):
        self.qnet.eval()
        self.test_mode = True

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        # load model weights
        if not self.test_mode:
            self.qnet.load_state_dict(checkpoint["qnet"])
        self.target_qnet.load_state_dict(checkpoint["qnet"])

    def resume(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        # load model weights
        self.qnet.load_state_dict(checkpoint["qnet"])
        self.target_qnet.load_state_dict(checkpoint["qnet"])
        self.opt.load_state_dict(checkpoint["opt"])

        self.start_episode = checkpoint["episode"] + 1

    def take_action(self, observation, **kwargs):
        if self.test_mode:
            return self.test_take_action(observation, **kwargs)
        else:
            return self.train_take_action(observation, **kwargs)

    def train_take_action(self, observation, **kwargs):
        if self.explorer.decide(observation):
            return self.explorer.act()
        else:
            with torch.no_grad():
                obs: Tensor = torch.tensor(
                    observation, dtype=torch.float, device=self.device
                ).view(-1, 1)

                action_dist: Tensor = self.qnet(obs).squeeze(0)

                action_index: int = action_dist.argmax().type(torch.int64).item()

            return action_index

    def test_take_action(self, observation, **kwargs):
        with torch.no_grad():
            obs: Tensor = torch.tensor(
                observation, dtype=torch.float, device=self.device
            ).view(-1, 1)

            action_dist: Tensor = self.qnet(obs).squeeze(0)

            action_index: int = action_dist.argmax().type(torch.int64).item()

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

        values_predict: Tensor = torch.gather(
            self.qnet(cur_observations),
            dim=1,
            index=cur_actions,
        )
        td_target = rewards + self.gamma * torch.max(
            self.target_qnet(next_observations),
            dim=1,
        )[0].view(-1, 1) * (1 - terminated)

        loss = torch.mean(F.mse_loss(values_predict, td_target))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if (logger.cur_episode_index + 1) % 10 == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

        return loss.item()

    def save(self, episode_index: int, save_path: str):
        checkpoint = dict(
            model_state_dict=self.qnet.state_dict(),
            optimizer_state_dict=self.opt.state_dict(),
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
                self.target_qnet(obs_indices),
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
