import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from labsurv.builders import AGENTS, STRATEGIES
from labsurv.models.agents import BaseAgent
from labsurv.runners.hooks import LoggerHook
from labsurv.utils.string import INDENT
from labsurv.utils.surveillance import generate_action_mask, info_room2actor_input
from numpy import ndarray as array
from numpy import pi as PI
from torch import Tensor
from torch.nn import Module


@AGENTS.register_module()
class OCPMultiAgentPPO(BaseAgent):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        actor_cfg: Dict,
        critic_cfg: Dict,
        device: Optional[str] = None,
        gamma: float = 0.9,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-4,
        update_step: int = 10,
        advantage_param: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_loss_coef: float = 0.01,
        agent_num: int = 1,
        pan_section_num: int = 360,
        tilt_section_num: int = 180,
        pan_range: List[float] = [-PI, PI],
        tilt_range: List[float] = [-PI / 2, PI / 2],
        cam_types: int = 1,
        mixed_reward: bool = False,
        backbone_path: Optional[str] = None,
        freeze_backbone: List[int] = [],
        load_from: Optional[str] = None,
        resume_from: Optional[str] = None,
        test_mode: bool = False,
    ):
        """
        The following combinations to specify arguments are allowed:
        1. `load_from`: train the agent from `load_from` with a new optimizer.
        2. `resume_from`: train the agent from `resume_from` with the exact optimizer
            `resume_from` was using.
        3. `load_from`, `test_mode`: test the agent from `load_from`.
        4. None of the above specified: train a brand new agent with a new optimizer.
        """

        if load_from is not None and resume_from is not None:
            raise ValueError(
                "`load_from` and `resume_from` should not be both specified."
            )
        if test_mode and load_from is None:
            raise ValueError("`load_from` should not be None in test mode.")
        if test_mode and resume_from is not None:
            raise ValueError(
                "Use `load_from` instead of `resume_from` to load model in test mode."
            )

        if (load_from is not None or resume_from is not None) and backbone_path is not None:
            raise ValueError(
                "`backbone_path` will be ignored if `load_from` "
                "or `resume_from` is not None"
            )

        super().__init__(device, gamma)

        self.test_mode = test_mode
        self.mixed_reward = mixed_reward
        self.cam_types = cam_types
        self.agent_num = agent_num

        self.pan_section_num = pan_section_num
        self.tilt_section_num = tilt_section_num
        self.pan_range = pan_range
        self.tilt_range = tilt_range

        self.actor: Module = STRATEGIES.build(actor_cfg).to(self.device)
        self.critic: Module = STRATEGIES.build(critic_cfg).to(self.device)

        if not self.test_mode:
            self.lr = [actor_lr, critic_lr]
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr[0])
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr[1])

            self.start_episode = 0
            self.update_step = update_step
            self.advantage_param = advantage_param
            self.clip_epsilon = clip_epsilon
            self.entropy_loss_coef = entropy_loss_coef

        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)
        elif backbone_path is not None:
            self.load_backbone(backbone_path, freeze_backbone)
        
        if len(freeze_backbone) > 0:
            freeze_name = tuple([f"set_abstraction.{index}" for index in freeze_backbone])
            for name, parameter in self.actor.backbone.named_parameters():
                if name.startswith(freeze_name):
                    parameter.requires_grad = False
            for name, parameter in self.critic.backbone.named_parameters():
                if name.startswith(freeze_name):
                    parameter.requires_grad = False

            self.actor_opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.actor.parameters()),
                lr=self.lr[0],
            )
            self.critic_opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.critic.parameters()),
                lr=self.lr[1],
            )


    def eval(self):
        self.test_mode = True
        self.actor.eval()

    def train(self):
        self.test_mode = False
        self.actor.train()

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.actor.load_state_dict(checkpoint["actor"]["model_state_dict"])
        self.critic.load_state_dict(checkpoint["critic"]["model_state_dict"])
        # One shall not load params of the optimizers, because learning rate
        # is contained in the state_dict of the optimizers, and loading
        # optimizer params will ignore the new learning rate.

    def resume(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        self.actor.load_state_dict(checkpoint["actor"]["model_state_dict"])
        self.critic.load_state_dict(checkpoint["critic"]["model_state_dict"])
        self.actor_opt.load_state_dict(checkpoint["actor"]["optimizer_state_dict"])
        self.critic_opt.load_state_dict(checkpoint["critic"]["optimizer_state_dict"])
        self.start_episode = checkpoint["episode"] + 1

    def load_backbone(self, backbone_path: str):
        """
        ## Description:

            Load pretrained params for PointNet++ backbone module.
        """
        backbone = torch.load(backbone_path)
        self.actor.backbone.load_state_dict(backbone)
        self.critic.backbone.load_state_dict(backbone)

    def take_action(
        self, room, cam_params: array, **kwargs
    ) -> Tuple[List[Tuple[array, array, array]], List[array], List[array]]:
        """
        ## Arguments:

            room (SurveillanceRoom).

            cam_params (np.ndarray): [AGENT_NUM, PARAM_DIM]. Absolute params of agents.

        ## Returns:

            observations (List[Tuple[array, array, array]]): [AGENT_NUM, Tuple], actor
            inputs.

            actions (List[array]): [AGENT_NUM, ACTION_DIM].
        """

        if self.test_mode:
            return self.test_take_action(room, cam_params, **kwargs)
        else:
            return self.train_take_action(room, cam_params, **kwargs)

    def train_take_action(
        self, room, cam_params: array, **kwargs
    ) -> Tuple[List[Tuple[array, array, array]], List[array], List[array]]:
        """
        ## Arguments:

            room (SurveillanceRoom).

            cam_params (np.ndarray): [AGENT_NUM, PARAM_DIM]. Absolute params of agents.

        ## Returns:

            observations (List[Tuple[array, array, array]]): [AGENT_NUM, Tuple], actor
            inputs.

            actions (List[array]): [AGENT_NUM, ACTION_DIM].

            action_masks (List[array]): [AGENT_NUM, ACTION_DIM].
        """

        logger: LoggerHook = kwargs["logger"]
        with torch.no_grad():  # if grad, memory leaks
            observations: List[Tuple[array, array, array]] = []
            actions: List[array] = []
            action_masks: List[array] = []

            log = None

            for agent_index, cam_param in enumerate(cam_params):
                actor_inputs = info_room2actor_input(room, self.agent_num, cam_param)
                observations.append(actor_inputs)

                self_and_neigh_params: Tensor = torch.tensor(
                    np.array([actor_inputs[0]]),
                    dtype=torch.float,
                    device=self.device,
                )  # [1, AGENT_NUM(NEIGH), PARAM_DIM]
                self_mask: Tensor = torch.tensor(  # [1, AGENT_NUM(NEIGH)]
                    np.array([actor_inputs[1]]),
                    dtype=torch.bool,
                    device=self.device,
                )
                neigh: Tensor = torch.tensor(  # [1, 3, 2L+1, 2L+1, 2L+1]
                    np.array([actor_inputs[2]]),
                    dtype=torch.float,
                    device=self.device,
                )

                # [ACTION_DIM]
                action_logits: Tensor = self.actor(
                    self_and_neigh_params, self_mask, neigh
                ).view(-1)
                action_mask = 1 - generate_action_mask(
                    room,
                    cam_param,
                    self.pan_section_num,
                    self.tilt_section_num,
                    self.pan_range,
                    self.tilt_range,
                )
                action_logits[action_mask == 1] = float("-inf")
                action_dist = F.softmax(action_logits, dim=-1)
                action_dist_cat = torch.distributions.Categorical(action_dist)

                # DEBUG(eric)
                log = _readable_action_dist(
                    kwargs["step_index"] + 1,
                    agent_index,
                    action_dist,
                    action_dist_cat.entropy().item(),
                    episode=kwargs["episode_index"] + 1,
                    action_mask=action_mask,
                    pred_log=log,
                )

                action_index = action_dist_cat.sample().type(torch.int64).item()

                actions.append(np.eye(len(action_dist))[[action_index]].reshape(-1))
                action_masks.append(action_mask)

        logger.show_log(
            _print_readable_action_dist(log, return_str=True), with_time=True, end=""
        )

        # [AGENT_NUM, Tuple], [AGENT_NUM, ACTION_DIM], [AGENT_NUM, ACTION_DIM]
        return observations, actions, action_masks

    def test_take_action(
        self, room, cam_params: array, **kwargs
    ) -> Tuple[List[Tuple[array, array, array]], List[array], List[array]]:
        """
        ## Arguments:

            room (SurveillanceRoom).

            cam_params (np.ndarray): [AGENT_NUM, PARAM_DIM]. Absolute params of agents.

        ## Returns:

            observations (List[Tuple[array, array, array]]): [AGENT_NUM, Tuple], actor
            inputs.

            actions (np.ndarray): [AGENT_NUM, ACTION_DIM].

            action_masks (List[array]): [AGENT_NUM, ACTION_DIM].
        """
        logger: LoggerHook = kwargs["logger"]

        with torch.no_grad():  # if grad, memory leaks
            observations: List[Tuple[array, array, array]] = []
            actions: List[array] = []
            action_masks: List[array] = []

            log = None

            for agent_index, cam_param in enumerate(cam_params):
                actor_inputs = info_room2actor_input(room, self.agent_num, cam_param)
                observations.append(actor_inputs)

                self_and_neigh_params: Tensor = torch.tensor(
                    np.array([actor_inputs[0]]),
                    dtype=torch.float,
                    device=self.device,
                )  # [1, AGENT_NUM(NEIGH), PARAM_DIM]
                self_mask: Tensor = torch.tensor(  # [1, AGENT_NUM(NEIGH)]
                    np.array([actor_inputs[1]]),
                    dtype=torch.bool,
                    device=self.device,
                )
                neigh: Tensor = torch.tensor(  # [1, 3, 2L+1, 2L+1, 2L+1]
                    np.array([actor_inputs[2]]),
                    dtype=torch.float,
                    device=self.device,
                )

                # [ACTION_DIM]
                action_logits: Tensor = self.actor(
                    self_and_neigh_params, self_mask, neigh
                ).view(-1)
                action_mask = 1 - generate_action_mask(
                    room,
                    cam_param,
                    self.pan_section_num,
                    self.tilt_section_num,
                    self.pan_range,
                    self.tilt_range,
                )
                action_logits[action_mask == 1] = float("-inf")
                action_dist = F.softmax(action_logits, dim=-1)
                action_dist_cat = torch.distributions.Categorical(action_dist)

                # DEBUG(eric)
                log = _readable_action_dist(
                    kwargs["step_index"] + 1,
                    agent_index,
                    action_dist,
                    action_dist_cat.entropy().item(),
                    action_mask=action_mask,
                    pred_log=log,
                )

                action_index = action_dist_cat.sample().type(torch.int64).item()

                actions.append(np.eye(len(action_dist))[[action_index]].reshape(-1))
                action_masks.append(action_mask)

        logger.show_log(
            _print_readable_action_dist(log, return_str=True), with_time=True, end=""
        )

        # [AGENT_NUM, Tuple], [AGENT_NUM, ACTION_DIM], [AGENT_NUM, ACTION_DIM]
        return observations, actions, action_masks

    def update(
        self,
        transitions: Dict[str, List[bool | float | array | Tuple[int, array]]],
        logger: LoggerHook,
    ) -> Tuple[float]:
        cur_observations: List[List[Tuple[array, array, array]]] = transitions[
            "cur_observation"
        ]  # [B, AGENT_NUM, Tuple]
        # [B, AGENT_NUM, ACTION_DIM]
        cur_actions: List[List[array]] = transitions["cur_action"]
        # [B, AGENT_NUM, ACTION_DIM]
        cur_action_masks: List[List[array]] = transitions["cur_action_mask"]
        cur_critic_inputs: List[Tuple[array, array]] = transitions["cur_critic_input"]
        next_critic_inputs: List[Tuple[array, array]] = transitions["next_critic_input"]
        rewards: List[float] = transitions["reward"]
        terminated: List[bool] = transitions["terminated"]

        # actor inputs
        (
            cur_self_and_neigh_params,  # [AGENT_NUM, B, AGENT_NUM(NEIGH), PARAM_DIM]
            cur_self_mask,  # [AGENT_NUM, B, AGENT_NUM(NEIGH)]
            cur_neigh,  # [AGENT_NUM, B, 3, 2L+1, 2L+1, 2L+1]
            cur_all_actions,  # [AGENT_NUM, B, ACTION_DIM]
            cur_all_action_masks,  # [AGENT_NUM, B, ACTION_DIM]
        ) = _reformat_actor_input(
            self.agent_num,
            self.device,
            cur_observations,
            cur_actions,
            cur_action_masks,
        )

        # critic inputs
        (
            cur_cam_params,  # [B, AGENT_NUM, PARAM_DIM]
            cur_envs,  # [B, 3, W, D, H]
            next_cam_params,  # [B, AGENT_NUM, PARAM_DIM]
            next_envs,  # [B, 3, W, D, H]
            mixed_rewards,  # [B, AGENT_NUM]
            system_rewards,  # [B]
            all_terminated,  # [B]
        ) = _reformat_critic_input(
            self.device,
            cur_critic_inputs,
            next_critic_inputs,
            rewards,
            terminated,
        )

        value_predict: Tensor = self.critic(next_cam_params, next_envs).view(-1)  # [B]
        critic_td_target: Tensor = system_rewards + self.gamma * value_predict * (
            1 - all_terminated
        )  # [B]
        if not self.mixed_reward:
            critic_td_error: Tensor = critic_td_target - self.critic(
                cur_cam_params, cur_envs
            )
            advantages: Tensor = _compute_advantage(  # [B]
                self.gamma, self.advantage_param, critic_td_error, self.device
            )

        for agent_index in range(self.agent_num):
            if self.mixed_reward:
                actor_td_target: Tensor = mixed_rewards[
                    :, agent_index
                ] + self.gamma * value_predict * (  # [B]
                    1 - all_terminated
                )
                actor_td_error: Tensor = actor_td_target - self.critic(
                    cur_cam_params, cur_envs
                )
                actor_advantages: Tensor = _compute_advantage(  # [B]
                    self.gamma, self.advantage_param, actor_td_error, self.device
                )
            cur_action_indices = (
                cur_all_actions[agent_index].nonzero()[:, 1].view(-1, 1)
            )  # [B, 1]
            pred_strat_prob = torch.log(  # [B]
                torch.gather(
                    F.softmax(
                        self.actor(
                            cur_self_and_neigh_params[agent_index],
                            cur_self_mask[agent_index],
                            cur_neigh[agent_index],
                        ),
                        dim=-1,
                    ),
                    dim=1,
                    index=cur_action_indices,
                ).view(-1)
            ).detach()

            for step in range(self.update_step):
                action_logits: Tensor = self.actor(  # [B, ACTION_DIM]
                    cur_self_and_neigh_params[agent_index],
                    cur_self_mask[agent_index],
                    cur_neigh[agent_index],
                )
                action_logits[cur_all_action_masks[agent_index] == 1] = float("-inf")
                action_dist = F.softmax(action_logits, dim=-1)

                cur_strat_prob = torch.log(  # [B]
                    torch.gather(
                        action_dist,
                        dim=1,
                        index=cur_action_indices,
                    ).view(-1)
                )

                entropy: Tensor = torch.distributions.Categorical(
                    action_dist
                ).entropy()  # [B]

                significance = torch.exp(cur_strat_prob - pred_strat_prob)  # [B]

                surrogate_1 = significance * (
                    actor_advantages if self.mixed_reward else advantages
                )  # [B]
                surrogate_2 = torch.clamp(  # [B]
                    significance,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon,
                ) * (
                    actor_advantages if self.mixed_reward else advantages
                )  # PPO clip

                actor_loss = torch.mean(  # PPO loss
                    -torch.min(surrogate_1, surrogate_2)
                )
                critic_loss = torch.mean(
                    F.mse_loss(
                        self.critic(cur_cam_params, cur_envs).view(-1),
                        critic_td_target.detach(),
                    )
                )
                entropy_loss = entropy.mean()
                total_loss = (
                    actor_loss
                    + 0.5 * critic_loss
                    - self.entropy_loss_coef * entropy_loss
                )

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                total_loss.backward()
                self.actor_opt.step()
                self.critic_opt.step()

                logger.show_log(
                    f"[Update step {step + 1:>3}  Agent {agent_index + 1}]  "
                    f"loss: C {critic_loss.item():.6f} "
                    f"A {actor_loss.item():.6f} E {entropy_loss.item():.6f}",
                    with_time=True,
                )

        return critic_loss.item(), actor_loss.item(), entropy_loss.item()

    def save(self, episode_index: int, save_path: str):
        checkpoint = dict(
            actor=dict(
                model_state_dict=self.actor.state_dict(),
                optimizer_state_dict=self.actor_opt.state_dict(),
            ),
            critic=dict(
                model_state_dict=self.critic.state_dict(),
                optimizer_state_dict=self.critic_opt.state_dict(),
            ),
            episode=episode_index,
        )

        episode = episode_index + 1
        if save_path.endswith(".pth"):
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            save_path = ".".join(save_path.split(".")[:-1]) + f"_episode_{episode}.pth"
        else:
            os.makedirs(save_path, exist_ok=True)
            save_path = osp.join(save_path, f"episode_{episode}.pth")

        torch.save(checkpoint, save_path)


def _readable_action_dist(
    step: int,
    agent_index: int,
    action_dist: Tensor,
    entropy: float,
    episode: Optional[int] = None,
    action_mask: Optional[Tensor] = None,
    pred_log: Optional[List[str | List[str]]] = None,
):
    if pred_log is None:
        pred_log = [
            (
                f"[Episode {episode:>4}  Step {step:>3}] action distributions"
                if episode is not None
                else f"[Step {step:>3}] action distributions"
            )
        ]

    cur_log = []
    cur_log.append(  # header line
        f"|{agent_index + 1:^7}{INDENT}{"x":^9}{INDENT}{"y":^9}{INDENT}"
        f"{"z":^9}{INDENT}{"p":^9}{INDENT}{"t":^9}{INDENT}"
    )

    action_probs = action_dist.clone().detach()
    if action_mask is not None:
        action_probs[action_mask == 1] = -1
    action_probs_list = [float(i) for i in action_probs.tolist()]
    cur_log.append(f"|{"-":^7}{INDENT}")  # "-" actions
    for i in range(5):
        cur_log[-1] += f"{action_probs_list[2 * i]:>9.6f}{INDENT}"
    cur_log.append(f"|{"+":^7}{INDENT}")  # "+" actions
    for i in range(5):
        cur_log[-1] += f"{action_probs_list[2 * i + 1]:>9.6f}{INDENT}"

    # cam lines
    cam_types = len(action_probs_list) - 10
    cam_line_count = int(np.ceil(cam_types / 5.0))
    for line in range(cam_line_count):
        cur_log.append(f"|{"cam":^7}{INDENT}")
        for i in range(5 * line, min(5 * (line + 1), cam_types)):
            cur_log[-1] += f"{i:^9}{INDENT}"
        for i in range(min(5 * (line + 1), cam_types), 5 * (line + 1)):
            cur_log[-1] += " " * 9 + INDENT

        cur_log.append("|" + " " * 7 + INDENT)
        for i in range(5 * line, min(5 * (line + 1), cam_types)):
            cur_log[-1] += f"{action_probs_list[10 + i]:>9.6f}{INDENT}"
        for i in range(min(5 * (line + 1), cam_types), 5 * (line + 1)):
            cur_log[-1] += " " * 9 + INDENT

    # entropy
    cur_log.append("|entropy" + INDENT + f"{entropy:>9.6f}" + INDENT)
    cur_log[-1] += (" " * 9 + INDENT) * 4

    if len(pred_log) == 1:
        for i in range(len(cur_log)):
            pred_log.append([cur_log[i]])
    else:
        for i in range(len(cur_log)):
            pred_log[i + 1].append(cur_log[i])

    return pred_log


def _print_readable_action_dist(
    action_dist_log: List[str | List[str]], step: int = 3, return_str: bool = False
):
    out_log = ""
    out_log += action_dist_log[0] + "\n"

    agent_num = len(action_dist_log[1])
    log_content = action_dist_log[1:]

    cur_agent_range = [0, min(step - 1, agent_num - 1)]
    while cur_agent_range[0] < agent_num:
        for line in log_content:
            for index in range(min(step, agent_num - cur_agent_range[0])):
                out_log += line[cur_agent_range[0] + index]
            out_log += "\n"
        cur_agent_range[0] += step
        cur_agent_range[1] += step

    if return_str:
        return out_log
    else:
        print(out_log, end="")


def _reformat_actor_input(
    agent_num: int,
    device: torch.cuda.device,
    cur_observations: List[List[Tuple[array, array, array]]],
    cur_actions: List[List[array]],
    cur_action_masks: List[List[array]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    ## permute batch channel and agent channel for actor inputs
    batch_size = len(cur_observations)
    cur_self_and_neigh_params_list: List[List[array]] = [[] for i in range(agent_num)]
    cur_self_mask_list: List[List[array]] = [[] for i in range(agent_num)]
    cur_neigh_list: List[List[array]] = [[] for i in range(agent_num)]
    cur_all_actions_list: List[List[array]] = [[] for i in range(agent_num)]
    cur_all_action_masks_list: List[List[array]] = [[] for i in range(agent_num)]
    for batch_index in range(batch_size):
        for agent_index in range(agent_num):
            cur_self_and_neigh_params_list[agent_index].append(
                cur_observations[batch_index][agent_index][0]
            )
            cur_self_mask_list[agent_index].append(
                cur_observations[batch_index][agent_index][1]
            )
            cur_neigh_list[agent_index].append(
                cur_observations[batch_index][agent_index][2]
            )
            cur_all_actions_list[agent_index].append(
                cur_actions[batch_index][agent_index]
            )
            cur_all_action_masks_list[agent_index].append(
                cur_action_masks[batch_index][agent_index]
            )
    # [AGENT_NUM, B, AGENT_NUM(NEIGH), PARAM_DIM]
    cur_self_and_neigh_params: Tensor = torch.tensor(
        np.array(cur_self_and_neigh_params_list),
        dtype=torch.float,
        device=device,
    )
    cur_self_mask: Tensor = torch.tensor(  # [AGENT_NUM, B, AGENT_NUM(NEIGH)]
        np.array(cur_self_mask_list), dtype=torch.bool, device=device
    )
    cur_neigh: Tensor = torch.tensor(  # [AGENT_NUM, B, 3, 2L+1, 2L+1, 2L+1]
        np.array(cur_neigh_list), dtype=torch.float, device=device
    )
    cur_all_actions: Tensor = torch.tensor(  # [AGENT_NUM, B, ACTION_DIM]
        np.array(cur_all_actions_list), dtype=torch.float, device=device
    )
    cur_all_action_masks: Tensor = torch.tensor(  # [AGENT_NUM, B, ACTION_DIM]
        np.array(cur_all_action_masks_list), dtype=torch.bool, device=device
    )

    return (
        cur_self_and_neigh_params,
        cur_self_mask,
        cur_neigh,
        cur_all_actions,
        cur_all_action_masks,
    )


def _reformat_critic_input(
    device: torch.cuda.device,
    cur_critic_inputs: List[Tuple[array, array]],
    next_critic_inputs: List[Tuple[array, array]],
    rewards: List[float],
    terminated: List[bool],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    cur_cam_params: Tensor = torch.tensor(  # [B, AGENT_NUM, PARAM_DIM]
        np.array([cur_critic_input[0] for cur_critic_input in cur_critic_inputs]),
        dtype=torch.float,
        device=device,
    )
    cur_envs: Tensor = torch.tensor(  # [B, 3, W, D, H]
        np.array([cur_critic_input[1] for cur_critic_input in cur_critic_inputs]),
        dtype=torch.float,
        device=device,
    )
    next_cam_params: Tensor = torch.tensor(  # [B, AGENT_NUM, PARAM_DIM]
        np.array([next_critic_input[0] for next_critic_input in next_critic_inputs]),
        dtype=torch.float,
        device=device,
    )
    next_envs: Tensor = torch.tensor(  # [B, 3, W, D, H]
        np.array([next_critic_input[1] for next_critic_input in next_critic_inputs]),
        dtype=torch.float,
        device=device,
    )
    all_rewards: Tensor = torch.tensor(  # [B, AGENT_NUM + 1]
        rewards, dtype=torch.float, device=device
    )
    mixed_rewards: Tensor = all_rewards[:, :-1]  # [B, AGENT_NUM]
    system_rewards: Tensor = all_rewards[:, -1]  # [B]
    all_terminated: Tensor = torch.tensor(  # [B]
        terminated, dtype=torch.int64, device=device
    )

    return (
        cur_cam_params,
        cur_envs,
        next_cam_params,
        next_envs,
        mixed_rewards,
        system_rewards,
        all_terminated,
    )


def _compute_advantage(
    gamma: float, advantage_param: float, td_error: Tensor, device: torch.cuda.device
):
    td_error = td_error.clone().detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0

    for delta in td_error[::-1]:
        advantage = gamma * advantage_param * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()

    return torch.tensor(
        np.array(advantage_list), dtype=torch.float, device=device
    )  # [B]
