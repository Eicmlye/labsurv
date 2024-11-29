import os.path as osp
from typing import Dict, Optional

from labsurv.builders import AGENTS, ENVIRONMENTS, HOOKS, REPLAY_BUFFERS, RUNNERS
from labsurv.models.agents import BaseAgent
from labsurv.models.buffers import BaseReplayBuffer, OCPPriorityReplayBuffer
from labsurv.models.envs import BaseSurveillanceEnv
from labsurv.runners.hooks import LoggerHook
from mmcv import Config
from mmcv.utils import ProgressBar


@RUNNERS.register_module()
class OCPStepBasedRunner:
    def __init__(self, cfg: Config):
        self.work_dir: str = cfg.work_dir
        self.logger: LoggerHook = HOOKS.build(cfg.logger_cfg)

        self.env: BaseSurveillanceEnv = ENVIRONMENTS.build(cfg.env)
        self.agent: BaseAgent = AGENTS.build(cfg.agent)
        self.test_mode: bool = self.agent.test_mode
        self.start_episode: int = 0

        # max step number for an episode, exceeding makes truncated True
        self.steps: int = cfg.steps

        if not self.test_mode:
            self.replay_buffer: Optional[BaseReplayBuffer] = None
            if cfg.use_replay_buffer:
                self.replay_buffer = REPLAY_BUFFERS.build(cfg.replay_buffer)
                if not self.replay_buffer.is_active():
                    self.generate_replay_buffer()

            self.save_checkpoint_interval: int = cfg.save_checkpoint_interval

            # max episode number
            self.episodes: int = cfg.episodes

            self.start_episode = self.agent.start_episode
            if "resume_from" in cfg.agent.keys() and cfg.agent.resume_from is not None:
                self.logger.set_cur_episode_index(self.start_episode)

    def generate_replay_buffer(self):
        cur_observation = None

        # generate replay buffer
        print("Replay buffer generating...")
        if self.replay_buffer is not None:
            prog_bar = ProgressBar(self.replay_buffer.activate_size)
            while not self.replay_buffer.is_active():
                cur_observation = self.env.reset()
                terminated = False

                for step in range(self.steps):
                    if terminated or self.replay_buffer.is_active():
                        break

                    cur_action_with_params = self.agent.take_action(
                        cur_observation, all_explore=True
                    )

                    transition, _, _ = self.env.step(
                        cur_observation, cur_action_with_params, self.steps
                    )
                    transition["cur_observation"] = cur_observation
                    transition["cur_action"] = cur_action_with_params

                    terminated = transition["terminated"]

                    if (
                        self.replay_buffer is not None
                        and not self.replay_buffer.is_active()
                    ):
                        self.replay_buffer.add(transition)
                        prog_bar.update()

                    cur_observation = transition["next_observation"]
        print("\nReplay buffer generated.")

        # save buffer cache
        self.replay_buffer.save(self.work_dir)

    def run(self):
        if self.test_mode:
            return self.test()
        else:
            return self.train()

    def train(self):
        cur_observation = None

        for episode in range(self.start_episode, self.episodes):
            cur_observation = self.env.reset()

            episode_return: Dict[str, float] = dict(
                critic_loss=0,
                actor_loss=0,
                reward=0,
            )
            terminated = False

            for step in range(self.steps):
                cur_action_with_params = self.agent.take_action(cur_observation)

                transition, cur_coverage, cam_count = self.env.step(
                    cur_observation, cur_action_with_params, self.steps
                )
                print(
                    f"{episode + 1} / {step + 1}: {cur_coverage * 100:.2f}% "
                    f"with {cam_count} cams & reward {transition["reward"]:.4f}"
                )
                transition["cur_observation"] = cur_observation
                transition["cur_action"] = cur_action_with_params

                terminated = transition["terminated"]
                # truncated = step == self.steps - 1

                if self.replay_buffer is not None:
                    self.replay_buffer.add(transition)

                cur_observation = transition["next_observation"]
                # add intrinsic reward
                transition["reward"] += self.agent.explorer.reward

                episode_return["reward"] += transition["reward"]

                if self.replay_buffer is not None:
                    if self.replay_buffer.is_active():
                        if isinstance(self.replay_buffer, OCPPriorityReplayBuffer):
                            self.replay_buffer.update(
                                self.agent.actor_target,
                                self.agent.critic_target,
                                self.agent.gamma,
                            )

                        samples = self.replay_buffer.sample()
                        (
                            episode_return["critic_loss"],
                            episode_return["actor_loss"],
                        ) = self.agent.update(samples)
                else:
                    raise NotImplementedError()

                if terminated:
                    break

            log_dict = dict(lr=self.agent.lr)

            self.logger.update(episode_return)
            self.logger(self.episodes, **log_dict)

            if (episode + 1) % self.save_checkpoint_interval == 0:
                self.agent.save(episode, self.work_dir)
                self.replay_buffer.save(
                    osp.join(self.work_dir, f"episode_{episode + 1}.pkl")
                )

    def test(self):
        cur_observation = self.env.reset()

        episode_return = []
        terminated = False

        print("Rendering test results...")
        prog_bar = ProgressBar(self.steps)
        for step in range(self.steps):
            self.env.render(cur_observation, step)

            cur_action = self.agent.take_action(cur_observation)

            transition = self.env.step(cur_observation, cur_action)
            transition["cur_observation"] = cur_observation
            transition["cur_action"] = cur_action

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
