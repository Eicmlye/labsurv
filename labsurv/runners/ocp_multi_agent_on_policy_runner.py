import os
import os.path as osp
import pickle
from copy import deepcopy
from typing import Dict, Optional

from labsurv.builders import AGENTS, ENVIRONMENTS, HOOKS, IMITATORS, RUNNERS
from labsurv.models.agents import BaseAgent
from labsurv.models.envs import BaseSurveillanceEnv
from labsurv.runners.hooks import LoggerHook
from labsurv.utils.string import get_time_stamp, readable_action, readable_param
from mmcv import Config
from numpy import ndarray as array


@RUNNERS.register_module()
class OCPMultiAgentOnPolicyRunner:
    def __init__(self, cfg: Config):
        self.work_dir: str = cfg.work_dir
        self.logger: LoggerHook = HOOKS.build(cfg.logger_cfg)

        self.env: BaseSurveillanceEnv = ENVIRONMENTS.build(cfg.env)
        self.agent: BaseAgent = AGENTS.build(cfg.agent)
        self.test_mode: bool = self.agent.test_mode
        self.start_episode: int = 0
        self.eval_interval: int = cfg.eval_interval

        # max step number for an episode, exceeding makes truncated True
        self.steps: int = cfg.steps

        if not self.test_mode:
            self.save_checkpoint_interval: int = cfg.save_checkpoint_interval

            # max episode number
            self.episodes: int = cfg.episodes

            self.start_episode = self.agent.start_episode
            if "resume_from" in cfg.agent.keys() and cfg.agent.resume_from is not None:
                self.logger.set_cur_episode_index(self.start_episode - 1)

            if hasattr(cfg, "expert") and cfg.expert is not None:
                self.expert = IMITATORS.build(cfg.expert)
                if not self.agent.mixed_reward:
                    raise RuntimeError(
                        "GAIL enabled and fake system reward is built. "
                        "Mixed rewards must be used. "
                    )

    def run(self):
        if self.test_mode:
            return self.test()
        else:
            return self.train()

    def train(self):
        cur_observation = None

        for episode in range(self.start_episode, self.episodes):
            self.logger.show_log(f"\n==== Episode {episode + 1} ====\n")
            # SurveillanceRoom, [AGENT_NUM, PARAM_DIM]
            room, cur_params = self.env.reset()

            episode_return: Dict[str, float] = dict(
                critic_loss=0,
                actor_loss=0,
                entropy_loss=0,
                reward=0,
                coverage=0,
            )
            terminated = False

            transitions = dict(
                cur_observation=[],
                cur_action=[],
                cur_action_mask=[],
                cur_critic_input=[],
                next_critic_input=[],
                reward=[],
                terminated=[],
            )

            for step in range(self.steps):
                cur_observation, cur_action, cur_action_mask = self.agent.take_action(
                    room,
                    cur_params,
                    episode_index=episode,
                    step_index=step,
                    save_dir=self.logger.save_dir,
                    logger=self.logger,
                )

                cur_coverage, cur_transition, new_params = self.env.step(
                    cur_observation, cur_action, self.steps, cur_action_mask
                )

                terminated = cur_transition["terminated"]
                truncated = step == self.steps - 1

                episode_return["reward"] += cur_transition["reward"][-1]
                episode_return["coverage"] = cur_coverage

                for key, item in cur_transition.items():
                    transitions[key].append(item)

                self.logger.show_log(
                    f"[Episode {episode + 1:>4} Step {step + 1:>3}]  {cur_coverage * 100:.2f}% "
                    f"| step reward {cur_transition["reward"][-1]:.4f} "
                    f"| total reward {episode_return["reward"]:.4f} "
                    f"\nPREVIOUS {_readable_param(cur_params)} "
                    f"\nACTION   {_readable_action(cur_action)} "
                    f"\nCURRENT  {_readable_param(new_params)} ",
                    with_time=True,
                )

                room = deepcopy(self.env.info_room)
                cur_params: array = new_params

                if terminated or truncated:
                    point_cloud_path = osp.join(
                        self.logger.save_dir,
                        "pointcloud",
                        "train",
                        f"epi{episode + 1}",
                        f"step{step + 1}_SurveillanceRoom_cam.ply",
                    )
                    self.env.info_room.visualize(
                        point_cloud_path, "camera", heatmap=True
                    )
                    break

            if hasattr(self, "expert"):
                expert_transitions = self.expert.sample()
                transitions = self.expert.train(transitions, expert_transitions)
            if self.agent.manual:
                expert_save_path = osp.join(self.logger.save_dir, "expert")
                os.makedirs(expert_save_path, exist_ok=True)
                with open(
                    osp.join(
                        expert_save_path,
                        f"{get_time_stamp()}_episode{episode + 1}_step{step + 1}.pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(transitions, f)
            else:
                (
                    episode_return["critic_loss"],
                    episode_return["actor_loss"],
                    episode_return["entropy_loss"],
                ) = self.agent.update(transitions, self.logger)
                self.logger.show_log(
                    f"episode reward {episode_return["reward"]:.4f} "
                    f"| loss: C {episode_return["critic_loss"]:.6f} "
                    f"A {episode_return["actor_loss"]:.6f} "
                    f"E {episode_return["entropy_loss"]:.6f} ",
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
                self.agent.save(episode, osp.join(self.work_dir, "models"))
                self.logger.show_log(f"Checkpoint saved at episode {episode + 1}.")
                self.env.save(episode, osp.join(self.work_dir, "envs"))
                self.logger.show_log(f"Environment saved at episode {episode + 1}.")
            if (episode + 1) % self.eval_interval == 0:
                self.agent.eval()
                self.test(episode)
                self.agent.train()

    def test(self, episode_index: Optional[int] = None):
        self.agent.eval()

        self.logger.show_log(
            "\nEvaluating agent"
            + ("" if episode_index is None else f" at episode {episode_index + 1} ")
            + "..."
        )

        # SurveillanceRoom, [AGENT_NUM, PARAM_DIM]
        room, cur_params = self.env.reset()

        episode_return: Dict[str, float] = dict(
            reward=0,
            coverage=0,
        )
        terminated = False

        for step in range(self.steps):
            cur_observation, cur_action, cur_action_mask = self.agent.take_action(
                room,
                cur_params,
                episode_index=episode_index,
                step_index=step,
                save_dir=self.logger.save_dir,
                logger=self.logger,
            )

            cur_coverage, cur_transition, new_params = self.env.step(
                cur_observation, cur_action, self.steps
            )

            terminated = cur_transition["terminated"]
            truncated = step == self.steps - 1

            episode_return["reward"] += cur_transition["reward"][-1]
            episode_return["coverage"] = cur_coverage

            self.logger.show_log(
                f"[Step {step + 1:>3}]  {cur_coverage * 100:.2f}% "
                f"| step reward {cur_transition["reward"][-1]:.4f} "
                f"| total reward {episode_return["reward"]:.4f} "
                f"\nCURRENT  {_readable_param(cur_params)} "
                f"\nACTION   {_readable_action(cur_action)} "
                f"\nPREVIOUS {_readable_param(new_params)} ",
                with_time=True,
            )

            room = deepcopy(self.env.info_room)
            cur_params: array = new_params

            point_cloud_path = osp.join(
                self.logger.save_dir,
                "pointcloud",
                "eval" if episode_index is not None else "test",
                (f"epi{episode_index + 1}/" if episode_index is not None else "")
                + f"step{step + 1}_SurveillanceRoom_cam.ply",
            )
            self.env.info_room.visualize(point_cloud_path, "camera", heatmap=True)

            if terminated or truncated:
                self.logger.show_log(
                    f"evaluation final reward {episode_return["reward"]:.4f}",
                    with_time=True,
                )
                break


def _readable_param(params: array) -> str:
    """
    ## Arguments:

        params (np.ndarray): [AGENT_NUM, PARAM_DIM]

    ## Returns:

        readable (str)
    """

    readable: str = ""

    for index, param in enumerate(params):
        readable += readable_param(param)
        if index < len(params) - 1:
            readable += ", "

    return readable


def _readable_action(actions: array):
    """
    ## Arguments:

        actions (np.ndarray): [AGENT_NUM, ACTION_DIM]

    ## Returns:

        readable (str)
    """

    readable: str = ""

    for index, action in enumerate(actions):
        readable += readable_action(action)
        if index < len(actions) - 1:
            readable += ", "

    return readable
