import os.path as osp
from typing import Dict

from labsurv.builders import AGENTS, ENVIRONMENTS, HOOKS, RUNNERS
from labsurv.models.agents import BaseAgent
from labsurv.models.envs import BaseSurveillanceEnv
from labsurv.runners.hooks import LoggerHook
from mmcv import Config
from mmcv.utils import ProgressBar
from numpy import pi as PI


@RUNNERS.register_module()
class OCPOnPolicyRunner:
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
            self.save_checkpoint_interval: int = cfg.save_checkpoint_interval

            # max episode number
            self.episodes: int = cfg.episodes

            self.start_episode = self.agent.start_episode
            if "resume_from" in cfg.agent.keys() and cfg.agent.resume_from is not None:
                self.logger.set_cur_episode_index(self.start_episode - 1)

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
                coverage=0,
            )
            terminated = False

            transitions = dict(
                cur_observation=[],
                cur_action=[],
                next_observation=[],
                reward=[],
                terminated=[],
            )

            for step in range(self.steps):
                cur_action_with_params = self.agent.take_action(cur_observation)

                section_nums = [
                    self.agent.pan_section_num,
                    self.agent.tilt_section_num,
                ]
                cur_coverage, cur_transition = self.env.step(
                    cur_observation, cur_action_with_params, self.steps, section_nums
                )

                terminated = cur_transition["terminated"]
                truncated = step == self.steps - 1

                cur_observation = cur_transition["next_observation"]

                episode_return["reward"] += cur_transition["reward"]
                episode_return["coverage"] = cur_coverage

                for key, item in transitions.items():
                    item.append(cur_transition[key])

                self.logger.show_log(
                    f"[Episode {episode + 1:>4} Step {step + 1:>3}]  {cur_coverage * 100:.2f}% "
                    f"| step reward {cur_transition["reward"]:.4f} "
                    f"| episode cur reward {episode_return["reward"]:.4f} "
                    f"\n\taction {int(cur_transition["cur_action"][0]):d} "
                    f"| pos [{int(cur_transition["cur_action"][1]):d}, "
                    f"{int(cur_transition["cur_action"][2]):d}, "
                    f"{int(cur_transition["cur_action"][3]):d}] "
                    f"| direction [{cur_transition["cur_action"][4] / 2 / PI * 360:.2f}, "
                    f"{cur_transition["cur_action"][5] / 2 / PI * 360:.2f}] "
                    f"| cam_type {int(cur_transition["cur_action"][6]):d}",
                    with_time=True,
                )

                if terminated or truncated:
                    point_cloud_path = osp.join(
                        self.logger.save_dir,
                        "pointcloud",
                        f"epi{episode + 1}_step{step + 1}_SurveillanceRoom_cam.ply",
                    )
                    self.env.info_room.visualize(point_cloud_path, "camera")
                    break

            (
                episode_return["critic_loss"],
                episode_return["actor_loss"],
            ) = self.agent.update(transitions, self.logger)
            self.logger.show_log(
                f"episode reward {episode_return["reward"]:.4f} "
                f"| loss: C {episode_return["critic_loss"]:.4f} "
                f"A {episode_return["actor_loss"]:.4f}",
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
