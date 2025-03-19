import math
from typing import Optional

import torch
from labsurv.builders import AGENTS, EXPLORERS
from labsurv.models.explorers import BaseExplorer
from torch import pi as PI


@AGENTS.register_module()
class BaseAgent:
    def __init__(
        self,
        device: str = None,
        gamma: float = 0.9,
        explorer_cfg: dict = None,
        load_from: Optional[str] = None,
        resume_from: Optional[str] = None,
        test_mode: bool = False,
    ):
        """
        ## Description:

            Base agent class for all agents.

        ## Arguments:

            device (str): the device the agent will work on.

            gamma (float): the discount factor of the agent.

            explorer_cfg (dict): the configuration for the explorer.

            load_from, resume_from (Optional[str]): The following combinations to
            specify arguments are allowed:
            1. `load_from`: train the agent from `load_from` with a new optimizer.
            2. `resume_from`: train the agent from `resume_from` with the exact
                optimizer `resume_from` was using.
            3. `load_from`, `test_mode`: test the agent from `load_from`.
            4. None of the above specified: train a brand new agent with a new
            optimizer.

            test_mode (bool): whether to use test mode.
        """

        if test_mode and load_from is None:
            raise ValueError("`load_from` should not be None in test mode.")
        if test_mode and resume_from is not None:
            raise ValueError("Use `load_from` to load model in test mode.")
        if load_from is not None and resume_from is not None:
            raise ValueError(
                "`load_from` and `resume_from` should not be both specified."
            )

        self.device = torch.device(device)
        self.gamma = gamma

        self.test_mode = test_mode

        if not self.test_mode:
            if explorer_cfg is not None:
                self.explorer: BaseExplorer = EXPLORERS.build(explorer_cfg)

            self.start_episode = 0

        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)

    def train(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def load(self, checkpoint_path: str):
        # checkpoint = torch.load(checkpoint_path)

        # load model weights
        # ...
        raise NotImplementedError()

    def resume(self, checkpoint_path: str):
        # checkpoint = torch.load(checkpoint_path)

        # load model weights
        # ...

        # self.start_episode = checkpoint["episode"]
        raise NotImplementedError()

    def take_action(self, observation):
        if self.test_mode:
            return self.test_take_action(observation)
        else:
            return self.train_take_action(observation)

    def train_take_action(self, observation):
        # if self.explorer.decide():
        #     return self.explorer.act()
        # else:
        #     action = None

        #     return action
        raise NotImplementedError()

    def test_take_action(self, observation):
        # action = None

        # return action
        raise NotImplementedError()

    def update(self, samples: dict):
        raise NotImplementedError()

    def update_scheduler(self, cur_episode: int, total_episode: int, mode: str = "cos"):
        """
        ## Description:

            Cosine one-cycle scheduler.
        """
        cur_episode = min(cur_episode, total_episode - 1)

        if mode == "cos":
            if isinstance(self.lr, list):
                for index in range(len(self.lr)):
                    time_index = PI / 4 + PI * 5 / 4 * cur_episode / (total_episode - 1)
                    min_lr = 1e-2 * self.max_lr[index]
                    amplification = (self.max_lr[index] - min_lr) / 2

                    self.lr[index] = amplification * (math.sin(time_index) + 1) + min_lr
            else:
                time_index = PI / 4 + PI * 5 / 4 * cur_episode / (total_episode - 1)
                min_lr = 1e-2 * self.max_lr
                amplification = (self.max_lr - min_lr) / 2

                self.lr = amplification * (math.sin(time_index) + 1) + min_lr
        else:
            raise NotImplementedError()

    def save(self, episode_index: int, save_path: str):
        # checkpoint = dict(
        #     model_state_dict=None,
        #     optimizer_state_dict=None,
        #     episode=episode_index,
        # )

        # episode = episode_index + 1
        # if save_path.endswith(".pth"):
        #     os.makedirs(osp.dirname(save_path), exist_ok=True)
        #     save_path = ".".join(save_path.split(".")[:-1]) + f"_episode_{episode}.pth"
        # else:
        #     os.makedirs(save_path, exist_ok=True)
        #     save_path = osp.join(save_path, f"episode_{episode}.pth")

        # torch.save(checkpoint, save_path)
        raise NotImplementedError()
