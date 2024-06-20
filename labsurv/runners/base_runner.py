import random

import numpy as np
from labsurv.builders import AGENTS, ENVIRONMENTS, HOOKS, REPLAY_BUFFERS
from mmengine import Config


class BaseRunner:
    def __init__(self, cfg: Config):
        np.random.seed(0)
        random.seed(0)

        self.env = ENVIRONMENTS.build(cfg.env)
        self.agent = AGENTS.build(cfg.agent)
        self.logger = HOOKS.build(cfg.logger_cfg)
        self.replay_buffer = (
            REPLAY_BUFFERS.build(cfg.replay_buffer) if cfg.use_replay_buffer else None
        )

        self.episodes = cfg.episodes

    def run(self):
        cur_observation = None
        for episode in range(self.episodes):
            cur_observation = self.env.reset()[0]

            episode_return = 0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                cur_action = self.agent.take_action(cur_observation)

                transitions = self.env.step(cur_action)
                transitions["cur_observation"] = cur_observation
                transitions["cur_action"] = cur_action

                if self.replay_buffer is not None:
                    self.replay_buffer.add(**transitions)

                cur_observation = transitions["next_observation"]
                episode_return += transitions["reward"]

                if self.replay_buffer is not None and self.replay_buffer.is_active():
                    transitions = self.replay_buffer.sample()
                    self.agent.update(**transitions)

            self.logger.update(episode_return)
            self.logger(self.episodes)
