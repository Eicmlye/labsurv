from labsurv.builders import AGENTS, ENVIRONMENTS, HOOKS, REPLAY_BUFFERS
from mmcv import Config


class EpisodeBasedRunner:
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
            markov_chain = dict()

            for step in range(self.steps):
                if terminated:
                    break

                cur_action = self.agent.take_action(cur_observation)

                transition = self.env.step(cur_observation, cur_action)
                transition["cur_observation"] = cur_observation
                transition["cur_action"] = cur_action

                terminated = transition["terminated"]

                cur_observation = transition["next_observation"]
                episode_return += transition["reward"]

                for key, val in transition.items():
                    if key not in markov_chain.keys():
                        markov_chain[key] = [val]
                    else:
                        markov_chain[key].append(val)

                # assert `transition` always has the same set of keys
                chain_item_len = -1
                for key, val in markov_chain.items():
                    if chain_item_len == -1:
                        chain_item_len = len(markov_chain[key])
                    else:
                        assert chain_item_len == len(markov_chain[key])

            self.agent.update(markov_chain)

            self.logger.update(episode_return)
            self.logger(self.episodes)
