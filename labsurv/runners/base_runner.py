from labsurv.builders import AGENTS, ENVIRONMENTS, HOOKS, REPLAY_BUFFERS
from mmengine import Config


class BaseRunner:
    def __init__(self, cfg: Config):
        self.env = ENVIRONMENTS.build(cfg.env)
        self.agent = AGENTS.build(cfg.agent)
        self.logger = HOOKS.build(cfg.logger_cfg)
        self.replay_buffer = (
            REPLAY_BUFFERS.build(cfg.replay_buffer) if cfg.use_replay_buffer else None
        )

        # max episode number
        self.episodes = cfg.episodes
        # max step number for an episode, exceeding makes truncated True
        self.steps = cfg.steps

    def run(self):
        cur_observation = None
        for episode in range(self.episodes):
            cur_observation = self.env.reset()

            episode_return = 0
            terminated = False

            for step in range(self.steps):
                if terminated:
                    break

                cur_action = self.agent.take_action(cur_observation)

                transition = self.env.step(cur_observation, cur_action)
                transition["cur_observation"] = cur_observation
                transition["cur_action"] = cur_action

                terminated = transition["terminated"]
                transition["truncated"] = step == self.steps - 1

                if self.replay_buffer is not None:
                    self.replay_buffer.add(transition)

                cur_observation = transition["next_observation"]
                episode_return += transition["reward"]

                if self.replay_buffer is not None and self.replay_buffer.is_active():
                    samples = self.replay_buffer.sample()
                    self.agent.update(samples)

            self.logger.update(episode_return)
            self.logger(self.episodes)
