import os.path as osp
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
from labsurv.builders import AGENTS, ENVIRONMENTS, HOOKS, RUNNERS
from labsurv.models.agents import BaseAgent
from labsurv.models.envs import BaseSurveillanceEnv
from labsurv.runners.hooks import LoggerHook
from labsurv.utils.string import INDENT
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
                )

                cur_coverage, cur_transition, new_params = self.env.step(
                    cur_observation, cur_action, self.steps, cur_action_mask
                )

                terminated = cur_transition["terminated"]
                truncated = step == self.steps - 1

                episode_return["reward"] += cur_transition["reward"][-1]
                episode_return["coverage"] = cur_coverage

                for key, item in transitions.items():
                    item.append(cur_transition[key])

                self.logger.show_log(
                    f"[Episode {episode + 1:>4} Step {step + 1:>3}]  {cur_coverage * 100:.2f}% "
                    f"| step reward {cur_transition["reward"][-1]:.4f} "
                    f"| total reward {episode_return["reward"]:.4f} "
                    f"\nPREVIOUS {readable_param(cur_params)} "
                    f"\nACTION   {readable_action(cur_action)} "
                    f"\nCURRENT  {readable_param(new_params)} ",
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
            cur_observation, cur_action = self.agent.take_action(
                room,
                cur_params,
                episode_index=episode_index,
                step_index=step,
                save_dir=self.logger.save_dir,
            )

            cur_coverage, cur_transition, new_params = self.env.step(
                cur_observation, cur_action, self.steps
            )

            terminated = cur_transition["terminated"]
            truncated = step == self.steps - 1

            room = deepcopy(self.env.info_room)
            cur_params: array = new_params

            episode_return["reward"] += cur_transition["reward"][-1]
            episode_return["coverage"] = cur_coverage

            self.logger.show_log(
                f"[Step {step + 1:>3}]  {cur_coverage * 100:.2f}% "
                f"| step reward {cur_transition["reward"][-1]:.4f} "
                f"| total reward {episode_return["reward"]:.4f} "
                f"\nCURRENT  {readable_param(cur_params)} "
                f"\nACTION   {readable_action(cur_action)} "
                f"\nPREVIOUS {readable_param(new_params)} ",
                with_time=True,
            )

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


def readable_param(params: array) -> str:
    """
    ## Arguments:

        params (np.ndarray): [AGENT_NUM, PARAM_DIM]

    ## Returns:

        readable (str)
    """

    readable: str = ""

    for index, param in enumerate(params):
        readable += _readable_param(param)
        if index < len(params) - 1:
            readable += ", "

    return readable


def _readable_param(param: array) -> str:
    """
    ## Arguments:

        param (np.ndarray): [PARAM_DIM]

    ## Returns:

        readable (str)
    """
    cache: List[array] = [
        param[:3].astype(np.int64),
        param[3:5].astype(np.float32),
        param[5:].nonzero()[0].astype(np.int64),
    ]

    cache[1] = cache[1] / np.pi * 180

    readable_list: List = []
    for i in range(len(cache)):
        readable_list += cache[i].tolist()

    readable: str = "["
    for i in range(3):
        readable += f"{readable_list[i]:>3d}" + INDENT
    for i in range(3, 5):
        readable += f"{readable_list[i]:>7.2f}" + INDENT
    readable += str(readable_list[-1]) + "]"

    return readable


def readable_action(actions: array):
    """
    ## Arguments:

        actions (np.ndarray): [AGENT_NUM, ACTION_DIM]

    ## Returns:

        readable (str)
    """

    readable: str = ""

    for index, action in enumerate(actions):
        readable += _readable_action(action)
        if index < len(actions) - 1:
            readable += ", "

    return readable


def _readable_action(action: array):
    """
    ## Arguments:

        action (np.ndarray): [ACTION_DIM]

    ## Returns:

        readable (str)
    """
    action_index: int = action.nonzero()[0].astype(np.int64)[0]

    if action_index < 2:
        readable = " " * 2 + ("-" if action_index % 2 == 0 else "+") + "x" + " " * 32
    elif action_index < 4:
        readable = " " * 7 + ("-" if action_index % 2 == 0 else "+") + "y" + " " * 27
    elif action_index < 6:
        readable = " " * 12 + ("-" if action_index % 2 == 0 else "+") + "z" + " " * 22
    elif action_index < 8:
        readable = " " * 21 + ("-" if action_index % 2 == 0 else "+") + "p" + " " * 13
    elif action_index < 10:
        readable = " " * 30 + ("-" if action_index % 2 == 0 else "+") + "t" + " " * 4
    else:
        readable = " " * 32 + "->" + str(action_index - 10)

    return readable
