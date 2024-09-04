import os.path as osp
from typing import Dict, List, Optional, Tuple

from labsurv.builders import AGENTS, ENVIRONMENTS, HOOKS, REPLAY_BUFFERS, RUNNERS
from labsurv.models.agents import OCPREINFORCE
from labsurv.models.buffers import BaseReplayBuffer
from labsurv.models.envs import BaseSurveillanceEnv
from labsurv.runners.hooks import LoggerHook
from mmcv import Config
from mmcv.utils import ProgressBar
from numpy import ndarray as array


@RUNNERS.register_module()
class OCPEpisodeBasedRunner:
    def __init__(self, cfg: Config):
        self.work_dir: str = cfg.work_dir
        self.logger: LoggerHook = HOOKS.build(cfg.logger_cfg)

        self.env: BaseSurveillanceEnv = ENVIRONMENTS.build(cfg.env)
        self.agent: OCPREINFORCE = AGENTS.build(cfg.agent)
        self.test_mode: bool = self.agent.test_mode
        self.start_episode: int = 0

        # max step number for an episode, exceeding makes truncated True
        self.steps = cfg.steps

        if not self.test_mode:
            self.replay_buffer = None
            if cfg.use_replay_buffer:
                self.replay_buffer: Optional[BaseReplayBuffer] = REPLAY_BUFFERS.build(
                    cfg.replay_buffer
                )
                if not self.replay_buffer.is_active():
                    self.generate_replay_buffer()

            self.save_checkpoint_interval: int = cfg.save_checkpoint_interval

            # max episode number
            self.episodes = cfg.episodes

            self.start_episode = self.agent.start_episode
            if "resume_from" in cfg.agent.keys() and cfg.agent.resume_from is not None:
                self.logger.set_cur_episode_index(self.start_episode)

    def generate_replay_buffer():
        pass

    def run(self):
        if self.test_mode:
            return self.test()
        else:
            return self.train()

    def train(self):
        cur_observation = None
        for episode in range(self.start_episode, self.episodes):
            print(f"Episode {episode + 1}:")
            cur_observation: array = self.env.reset()  # [12, W, D, H]

            episode_return: Dict[str, float] = dict(
                loss=0,
                reward=0,
                final_cov=0,
                max_cov=0,
                final_cam_count=0,
                max_cam_count=0,
            )
            terminated: bool = False
            markov_chain: Dict[str, List[array | float | Tuple[array, array]]] = dict()

            for step in range(self.steps):
                print(f"Step {step + 1}:")
                cur_action, params = self.agent.take_action(cur_observation)

                (
                    transition,
                    episode_return["final_cov"],
                    episode_return["final_cam_count"],
                ) = self.env.step(cur_observation, cur_action, params, self.steps)
                episode_return["max_cov"] = max(
                    episode_return["final_cov"], episode_return["max_cov"]
                )
                episode_return["max_cam_count"] = max(
                    episode_return["final_cam_count"], episode_return["max_cam_count"]
                )

                transition["cur_observation"] = cur_observation
                transition["cur_action"] = (cur_action, params)

                terminated = transition["terminated"]
                # truncated = step == self.steps - 1

                cur_observation = transition["next_observation"]
                episode_return["reward"] += transition["reward"]

                for key, val in transition.items():
                    if key not in markov_chain.keys():
                        markov_chain[key] = [val]
                    else:
                        markov_chain[key].append(val)

                print("\r\033[1A\033[K", end="")

                if terminated:
                    break

            episode_return["loss"] = self.agent.update(markov_chain)

            print("\r\033[1A\033[K", end="")

            log_dict = dict(lr=self.agent.lr)

            self.logger.update(episode_return)
            self.logger(self.episodes, **log_dict)

            if (episode + 1) % self.save_checkpoint_interval == 0:
                self.agent.save(episode, self.work_dir)

    def test(self):
        cur_observation: array = self.env.reset()  # [12, W, D, H]

        episode_return = []
        terminated: bool = False

        print("Rendering test results...")
        prog_bar = ProgressBar(self.steps)
        for step in range(self.steps):
            self.env.render(cur_observation, step)

            cur_action, params = self.agent.take_action(cur_observation)

            transition, _, _ = self.env.step(cur_observation, cur_action, params)
            transition["cur_observation"] = cur_observation
            transition["cur_action"] = (cur_action, params)

            terminated = transition["terminated"]

            cur_observation = transition["next_observation"]
            episode_return.append(transition["reward"])

            prog_bar.update()

            if terminated:
                break

        if hasattr(self.env, "save_gif"):
            print("\nSaving gif...")
            save_path = osp.join(self.work_dir, f"done_step_{prog_bar.completed}.gif")
            self.env.save_gif(save_path)
            print(f"GIF saved to {save_path}")
