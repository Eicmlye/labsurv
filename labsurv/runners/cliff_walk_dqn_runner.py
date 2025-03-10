import os.path as osp
from typing import Dict, Optional

from labsurv.builders import AGENTS, ENVIRONMENTS, HOOKS, REPLAY_BUFFERS, RUNNERS
from labsurv.runners.hooks import LoggerHook
from mmcv import Config
from mmcv.utils import ProgressBar


@RUNNERS.register_module()
class CliffWalkDQNRunner:
    def __init__(self, cfg: Config):
        self.work_dir: str = cfg.work_dir
        self.logger: LoggerHook = HOOKS.build(cfg.logger_cfg)

        self.env = ENVIRONMENTS.build(cfg.env)
        self.agent = AGENTS.build(cfg.agent)
        self.test_mode: bool = self.agent.test_mode
        self.start_episode: int = 0
        self.eval_interval: int = cfg.eval_interval

        # max step number for an episode, exceeding makes truncated True
        self.steps: int = cfg.steps

        if not self.test_mode:
            self.replay_buffer = REPLAY_BUFFERS.build(cfg.replay_buffer)
            if not self.replay_buffer.is_active():
                self.generate_replay_buffer()

            self.save_checkpoint_interval: int = cfg.save_checkpoint_interval

            # max episode number
            self.episodes: int = cfg.episodes

            self.start_episode = self.agent.start_episode
            if "resume_from" in cfg.agent.keys() and cfg.agent.resume_from is not None:
                self.logger.set_cur_episode_index(self.start_episode - 1)

    def generate_replay_buffer(self):
        cur_observation = None

        if self.replay_buffer is None:
            raise RuntimeError("No replay buffer detected.")

        # generate replay buffer
        print("Replay buffer generating...")
        prog_bar = ProgressBar(self.replay_buffer.activate_size)
        while not self.replay_buffer.is_active():
            cur_observation = self.env.reset()
            terminated = False

            for step in range(self.steps):
                if terminated or self.replay_buffer.is_active():
                    break

                cur_action = self.agent.take_action(cur_observation, all_explore=True)

                cur_transition = self.env.step(cur_observation, cur_action)

                terminated = cur_transition["terminated"]

                self.replay_buffer.add(cur_transition)
                prog_bar.update()

                cur_observation = cur_transition["next_observation"]
        print("\nReplay buffer generated.")

        self.agent.explorer.reset_epsilon()

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
            self.logger.show_log(f"\n==== Episode {episode + 1} ====\n")

            cur_observation = self.env.reset()

            episode_return: Dict[str, float] = dict(
                loss=0,
                reward=0,
                coverage=0,
            )
            terminated = False

            for step in range(self.steps):
                cur_action = self.agent.take_action(
                    cur_observation,
                    episode_index=episode,
                    step_index=step,
                    logger=self.logger,
                    save_dir=self.logger.save_dir,
                )

                cur_transition = self.env.step(cur_observation, cur_action)

                cur_observation = cur_transition["next_observation"]
                terminated = cur_transition["terminated"]
                truncated = step == self.steps - 1

                episode_return["reward"] += cur_transition["reward"]

                self.replay_buffer.add(cur_transition)

                self.logger.show_log(
                    f"[Episode {episode + 1:>4} Step {step + 1:>3}]  "
                    f"step reward {cur_transition["reward"]:.4f} "
                    f"| total reward {episode_return["reward"]:.4f} ",
                    with_time=True,
                )

                if terminated or truncated:
                    break

            self.replay_buffer.update(
                self.agent.qnet, self.agent.target_qnet, self.agent.gamma
            )
            episode_return["loss"] = self.agent.update(
                self.replay_buffer.sample(), self.logger
            )
            self.logger.show_log(
                f"episode reward {episode_return["reward"]:.4f} "
                f"| loss {episode_return["loss"]:.4f}",
                with_time=True,
            )

            log_dict = (
                dict(
                    actor_lr=self.agent.lr[0],
                    critic_lr=self.agent.lr[1],
                )
                if isinstance(self.agent.lr, list)
                else dict(lr=self.agent.lr)
            )

            self.logger.update(episode_return)
            self.logger(self.episodes, **log_dict)

            if (episode + 1) % self.save_checkpoint_interval == 0:
                self.agent.save(episode, self.work_dir)
                self.logger.show_log(f"Checkpoint saved at episode {episode + 1}.")
                self.replay_buffer.save(
                    osp.join(self.work_dir, f"replay_buffer_{episode + 1}.pkl")
                )
                self.logger.show_log(f"Replay buffer saved at episode {episode + 1}.")
            if (episode + 1) % self.eval_interval == 0:
                self.agent.eval()
                self.test(episode)
                self.agent.print_strat(self.env.env, self.logger)
                self.agent.train()

            self.logger.show_log(str(self.env.env_count))

    def test(self, episode_index: Optional[int] = None):
        self.agent.eval()

        self.logger.show_log(
            "\nEvaluating agent"
            + ("" if episode_index is None else f" at episode {episode_index + 1} ")
            + "..."
        )

        cur_observation = self.env.reset(test_mode=True)

        episode_return: Dict[str, float] = dict(
            reward=0,
            coverage=0,
        )

        for step in range(self.steps):
            cur_action = self.agent.take_action(
                cur_observation,
                episode_index=episode_index,
                step_index=step,
                logger=self.logger,
                save_dir=self.logger.save_dir,
            )

            cur_transition = self.env.step(cur_observation, cur_action)

            cur_observation = cur_transition["next_observation"]
            terminated = cur_transition["terminated"]
            truncated = step == self.steps - 1

            episode_return["reward"] += cur_transition["reward"]

            self.logger.show_log(
                f"[Step {step + 1:>3}]  "
                f"step reward {cur_transition["reward"]:.4f} "
                f"| episode cur reward {episode_return["reward"]:.4f} ",
                with_time=True,
            )

            if terminated or truncated:
                self.logger.show_log(
                    f"evaluation final reward {episode_return["reward"]:.4f}",
                    with_time=True,
                )
                break
