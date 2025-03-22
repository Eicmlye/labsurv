import os
import os.path as osp
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from labsurv.builders import AGENTS, REPLAY_BUFFERS, STRATEGIES
from labsurv.models.agents import BaseAgent
from labsurv.models.buffers import OCPReplayBuffer
from labsurv.runners.hooks import LoggerHook
from labsurv.utils.string import INDENT, readable_param
from labsurv.utils.surveillance import (
    generate_action_mask,
    info_room2actor_input,
    reformat_input,
)
from mmcv.utils import ProgressBar
from numpy import ndarray as array
from numpy import pi as PI
from torch import Tensor
from torch.nn import Module


@AGENTS.register_module()
class OCPMultiAgentGRPO(BaseAgent):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        actor_cfg: Dict,
        buffer_cfg: Dict,
        device: Optional[str] = None,
        gamma: float = 0.9,
        gradient_accumulation_batchsize: Optional[int] = None,
        actor_lr: float = 1e-5,
        update_step: int = 10,
        advantage_param: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_loss_coef: float = 0.01,
        agent_num: int = 2,
        pan_section_num: int = 360,
        tilt_section_num: int = 180,
        pan_range: List[float] = [-PI, PI],
        tilt_range: List[float] = [-PI / 2, PI / 2],
        allow_polar: bool = False,
        cam_types: int = 1,
        mixed_reward: bool = False,
        backbone_path: Optional[str] = None,
        freeze_backbone: List[int] = [],
        load_from: Optional[str] = None,
        resume_from: Optional[str] = None,
        test_mode: bool = False,
        manual: bool = False,
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

        if (
            load_from is not None or resume_from is not None
        ) and backbone_path is not None:
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
        self.allow_polar = allow_polar

        self.actor: Module = STRATEGIES.build(actor_cfg).to(self.device)
        self.manual = manual

        if not self.test_mode:
            self.buffer: OCPReplayBuffer = REPLAY_BUFFERS.build(buffer_cfg)

            self.gradient_accumulation_batchsize = (
                gradient_accumulation_batchsize
                if gradient_accumulation_batchsize is not None
                else self.agent_num
            )
            self.max_lr = actor_lr
            self.lr = actor_lr

            self.start_episode = 0
            self.update_step = update_step
            self.advantage_param = advantage_param
            self.clip_epsilon = clip_epsilon
            self.entropy_loss_coef = entropy_loss_coef

            if len(freeze_backbone) > 0:
                freeze_name = tuple(
                    [f"set_abstraction.{index}" for index in freeze_backbone]
                )
                for name, parameter in self.actor.backbone.named_parameters():
                    if name.startswith(freeze_name):
                        parameter.requires_grad = False
                self.actor_opt = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.actor.parameters()),
                    lr=self.lr[0],
                )
            else:
                self.actor_opt = torch.optim.Adam(
                    self.actor.parameters(), lr=self.lr[0]
                )

        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)
        elif backbone_path is not None:
            self.load_backbone(backbone_path)

    def eval(self):
        self.test_mode = True
        self.actor.eval()

    def train(self):
        self.test_mode = False
        self.actor.train()

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        missing_keys, unexpected_keys = self.actor.load_state_dict(
            checkpoint["actor"]["model_state_dict"], strict=False
        )
        if len(missing_keys) > 0:
            print(f"Missing keys in actor checkpoint: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys in actor checkpoint: {unexpected_keys}")

        # One shall not load params of the optimizers, because learning rate
        # is contained in the state_dict of the optimizers, and loading
        # optimizer params will ignore the new learning rate.

    def resume(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        missing_keys, unexpected_keys = self.actor.load_state_dict(
            checkpoint["actor"]["model_state_dict"], strict=False
        )
        if len(missing_keys) > 0:
            print(f"Missing keys in actor checkpoint: \n{missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys in actor checkpoint: \n{unexpected_keys}")

        self.actor_opt.load_state_dict(checkpoint["actor"]["optimizer_state_dict"])

        self.start_episode = checkpoint["episode"] + 1

    def load_backbone(self, backbone_path: str):
        """
        ## Description:

            Load pretrained params for PointNet++ backbone module.
        """
        backbone = torch.load(backbone_path)
        self.actor.backbone.load_state_dict(backbone)

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
        elif self.manual:
            return self.manual_take_action(room, cam_params, **kwargs)
        else:
            return self.train_take_action(room, cam_params, **kwargs)

    def manual_take_action(
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

        # logger: LoggerHook = kwargs["logger"]
        with torch.no_grad():  # if grad, memory leaks
            observations: List[Tuple[array, array, array]] = []
            actions: List[array] = []
            action_masks: List[array] = []

            for agent_index, cam_param in enumerate(cam_params):
                actor_inputs = info_room2actor_input(room, self.agent_num, cam_param)
                observations.append(actor_inputs)

                action_mask = 1 - generate_action_mask(
                    room,
                    cam_param,
                    self.pan_section_num,
                    self.tilt_section_num,
                    self.pan_range,
                    self.tilt_range,
                    allow_polar=self.allow_polar,
                )

                room.visualize(
                    osp.join(
                        kwargs["save_dir"],
                        "pointcloud",
                        "manual",
                        f"agent{agent_index + 1}_full_room.ply",
                    ),
                    "camera",
                    heatmap=True,
                    emphasis=torch.tensor(cam_param[:3]),
                )

                input_action: str = input(
                    f"Agent {agent_index + 1}: {readable_param(cam_param)}"
                    "\nChoose action:  x  y  z  p  t"
                    "\n             -  A  S  .  E  F"
                    "\n             +  D  W  .  Q  R"
                    "\nYour action: "
                )
                while (
                    len(input_action) == 0 or input_action[0].upper() not in "ADSWEQFR"
                ):
                    input_action = input("Illegal input. Your action: ")
                input_dict = dict(A=0, D=1, S=2, W=3, E=6, Q=7, F=8, R=9)

                actions.append(
                    np.eye(self.actor.out[2].out_features)[
                        input_dict[input_action[0].upper()]
                    ].reshape(-1)
                )
                action_masks.append(action_mask)

        # [AGENT_NUM, Tuple], [AGENT_NUM, ACTION_DIM], [AGENT_NUM, ACTION_DIM]
        return observations, actions, action_masks

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
                    allow_polar=self.allow_polar,
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
                    allow_polar=self.allow_polar,
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

    def _add_to_buffer(self, transitions: Dict):
        actor_inputs, critic_inputs = reformat_input(
            self.agent_num, self.device, transitions
        )

        # actor inputs
        (
            cur_self_and_neigh_params,  # [AGENT_NUM, B, AGENT_NUM(NEIGH), PARAM_DIM]
            cur_self_mask,  # [AGENT_NUM, B, AGENT_NUM(NEIGH)]
            cur_neigh,  # [AGENT_NUM, B, 3, 2L+1, 2L+1, 2L+1]
            cur_all_actions,  # [AGENT_NUM, B, ACTION_DIM]
            cur_all_action_masks,  # [AGENT_NUM, B, ACTION_DIM]
            _,
            _,
            _,
        ) = actor_inputs

        batch_size: int = cur_self_and_neigh_params.shape[1]
        param_dim: int = cur_self_and_neigh_params.shape[3]
        neigh_side_length: int = cur_neigh.shape[3]

        # critic inputs
        (
            cur_cam_params,  # [B, AGENT_NUM, PARAM_DIM]
            cur_envs,  # [B, 3, W, D, H]
            next_cam_params,  # [B, AGENT_NUM, PARAM_DIM]
            next_envs,  # [B, 3, W, D, H]
            mixed_rewards,  # [B, AGENT_NUM]
            system_rewards,  # [B]
            all_terminated,  # [B]
        ) = critic_inputs

        self.buffer.add(
            dict(
                # [AGENT_NUM, B, AGENT_NUM(NEIGH), PARAM_DIM]
                cur_self_and_neigh_params=cur_self_and_neigh_params,
                cur_self_mask=cur_self_mask,  # [AGENT_NUM, B, AGENT_NUM(NEIGH)]
                cur_neigh=cur_neigh,  # [AGENT_NUM, B, 3, 2L+1, 2L+1, 2L+1]
                cur_all_actions=cur_all_actions,  # [AGENT_NUM, B, ACTION_DIM]
                # [AGENT_NUM, B, ACTION_DIM]
                cur_all_action_masks=cur_all_action_masks,
                cur_cam_params=cur_cam_params,  # [B, AGENT_NUM, PARAM_DIM]
                cur_envs=cur_envs,  # [B, 3, W, D, H]
                next_cam_params=next_cam_params,  # [B, AGENT_NUM, PARAM_DIM]
                next_envs=next_envs,  # [B, 3, W, D, H]
                mixed_rewards=mixed_rewards,  # [B, AGENT_NUM]
                system_rewards=system_rewards,  # [B]
                all_terminated=all_terminated,  # [B]
            )
        )

    def update(
        self,
        transitions: Dict[str, List[bool | float | array | Tuple[int, array]]],
        logger: LoggerHook,
    ) -> Tuple[float]:
        self._add_to_buffer(transitions)

        if not self.buffer.is_active():
            logger.show_log(
                f"[Update step {step + 1:>3}]  " "loss: C 0.0 A 0.0 E 0.0",
                with_time=True,
            )

            return 0.0, 0.0, 0.0

        tracks: Dict[str, List[Tensor]] = self.buffer.sample()
        cur_self_and_neigh_params = torch.tensor(
            tracks["cur_self_and_neigh_params"], dtype=torch.float, device=self.device
        )  # [EPISODE, AGENT_NUM, B, AGENT_NUM(NEIGH), PARAM_DIM]
        cur_self_mask = torch.tensor(  # [EPISODE, AGENT_NUM, B, AGENT_NUM(NEIGH)]
            tracks["cur_self_mask"], dtype=torch.float, device=self.device
        )
        cur_neigh = torch.tensor(
            tracks["cur_neigh"], dtype=torch.float, device=self.device
        )  # [EPISODE, AGENT_NUM, B, 3, 2L+1, 2L+1, 2L+1]
        cur_all_actions = torch.tensor(  # [EPISODE, AGENT_NUM, B, ACTION_DIM]
            tracks["cur_all_actions"], dtype=torch.float, device=self.device
        )
        cur_all_action_masks = torch.tensor(  # [EPISODE, AGENT_NUM, B, ACTION_DIM]
            tracks["cur_all_action_masks"], dtype=torch.float, device=self.device
        )
        cur_cam_params = torch.tensor(  # [EPISODE, B, AGENT_NUM, PARAM_DIM]
            tracks["cur_cam_params"], dtype=torch.float, device=self.device
        )
        cur_envs = torch.tensor(  # [EPISODE, B, 3, W, D, H]
            tracks["cur_envs"], dtype=torch.float, device=self.device
        )
        next_cam_params = torch.tensor(  # [EPISODE, B, AGENT_NUM, PARAM_DIM]
            tracks["next_cam_params"], dtype=torch.float, device=self.device
        )
        next_envs = torch.tensor(  # [EPISODE, B, 3, W, D, H]
            tracks["next_envs"], dtype=torch.float, device=self.device
        )
        mixed_rewards = torch.tensor(  # [EPISODE, B, AGENT_NUM]
            tracks["mixed_rewards"], dtype=torch.float, device=self.device
        )
        system_rewards = torch.tensor(  # [EPISODE, B]
            tracks["system_rewards"], dtype=torch.float, device=self.device
        )
        all_terminated = torch.tensor(  # [EPISODE, B]
            tracks["all_terminated"], dtype=torch.int64, device=self.device
        )

        batch_size: int = cur_self_and_neigh_params[0].shape[1]
        param_dim: int = cur_self_and_neigh_params[0].shape[3]
        neigh_side_length: int = cur_neigh[0].shape[3]

        if not self.mixed_reward:
            advantages: Tensor = _outcome_supervision(  # [AGENT_NUM * B]
                self.gamma, self.advantage_param, system_rewards, self.device
            ).repeat(self.agent_num)
        else:
            all_agents_actor_advantages: Tensor = None  # [AGENT_NUM * B]
            for agent_index in range(self.agent_num):
                actor_advantages: Tensor = _process_supervision(  # [B]
                    self.gamma, self.advantage_param, mixed_rewards, self.device
                )

                if all_agents_actor_advantages is None:
                    all_agents_actor_advantages = actor_advantages.clone()
                else:
                    all_agents_actor_advantages = torch.cat(
                        (all_agents_actor_advantages, actor_advantages),
                        dim=0,
                    )

        cur_action_indices = cur_all_actions.nonzero()[:, -1].view(-1, 1)  # [B, 1]

        pred_actor = deepcopy(self.actor)

        gradient_accumulation_batchnum = int(
            np.ceil(batch_size * self.agent_num / self.gradient_accumulation_batchsize)
        )
        for param_group in self.actor_opt.param_groups:
            param_group["lr"] = self.lr / gradient_accumulation_batchnum

        for step in range(self.update_step):
            self.actor_opt.zero_grad()
            print(f"\nGradient accumulation for step {step + 1}...")
            prog_bar = ProgressBar(gradient_accumulation_batchnum)
            for ga_step in range(gradient_accumulation_batchnum):
                lower_index = ga_step * self.gradient_accumulation_batchsize
                upper_index_excluded = min(
                    (ga_step + 1) * self.gradient_accumulation_batchsize,
                    batch_size * self.agent_num,
                )

                ga_cur_self_and_neigh_params = cur_self_and_neigh_params.view(
                    batch_size * self.agent_num, -1, param_dim
                )[lower_index:upper_index_excluded]
                ga_cur_self_mask = cur_self_mask.view(batch_size * self.agent_num, -1)[
                    lower_index:upper_index_excluded
                ]
                ga_cur_neigh = cur_neigh.view(
                    batch_size * self.agent_num,
                    3,
                    neigh_side_length,
                    neigh_side_length,
                    neigh_side_length,
                )[lower_index:upper_index_excluded]

                action_logits: Tensor = self.actor(  # [GA, ACTION_DIM]
                    ga_cur_self_and_neigh_params, ga_cur_self_mask, ga_cur_neigh
                )
                action_logits[
                    cur_all_action_masks.view(batch_size * self.agent_num, -1)[
                        lower_index:upper_index_excluded
                    ]
                    == 1
                ] = float("-inf")
                action_dist = F.softmax(action_logits, dim=-1)  # [GA, ACTION_DIM]

                cur_strat_prob = torch.log(  # [GA]
                    torch.gather(
                        action_dist + 1e-8,
                        dim=1,
                        index=cur_action_indices[lower_index:upper_index_excluded],
                    ).view(-1)
                )

                entropy: Tensor = torch.distributions.Categorical(  # [GA]
                    action_dist
                ).entropy()

                pred_strat_prob = torch.log(  # [GA]
                    torch.gather(
                        F.softmax(
                            pred_actor(
                                ga_cur_self_and_neigh_params,
                                ga_cur_self_mask,
                                ga_cur_neigh,
                            ),
                            dim=-1,
                        )
                        + 1e-8,
                        dim=1,
                        index=cur_action_indices[lower_index:upper_index_excluded],
                    ).view(-1)
                ).detach()

                significance = torch.exp(cur_strat_prob - pred_strat_prob)  # [GA]

                surrogate_1 = significance * (  # [GA]
                    all_agents_actor_advantages[lower_index:upper_index_excluded]
                    if self.mixed_reward
                    else advantages[lower_index:upper_index_excluded]
                )
                surrogate_2 = torch.clamp(  # [GA]
                    significance,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon,
                ) * (
                    all_agents_actor_advantages[lower_index:upper_index_excluded]
                    if self.mixed_reward
                    else advantages[lower_index:upper_index_excluded]
                )  # PPO clip

                actor_loss = torch.mean(  # PPO loss
                    -torch.min(surrogate_1, surrogate_2)
                )
                entropy_loss = entropy.mean()
                total_loss = (
                    actor_loss - self.entropy_loss_coef * entropy_loss
                ) / gradient_accumulation_batchnum

                total_loss.backward()

                prog_bar.update()
            print("\r\033[K\033[1A\033[K", end="")

            self.actor_opt.step()

            logger.show_log(
                f"[Update step {step + 1:>3}]  "
                "loss: C 0.0 "
                f"A {actor_loss.item():.10f} E {entropy_loss.item():.10f}",
                with_time=True,
            )

        return 0, actor_loss.item(), entropy_loss.item()

    def save(self, episode_index: int, save_path: str):
        checkpoint = dict(
            actor=dict(
                model_state_dict=self.actor.state_dict(),
                optimizer_state_dict=self.actor_opt.state_dict(),
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


def _outcome_supervision(
    gamma: float, advantage_param: float, rewards: Tensor, device: torch.cuda.device
):
    """
    ## Arguments:

        gamma (float)

        advantage_param (float)

        rewards (Tensor): [EPISODE, B]

        device (torch.cuda.device)
    """
    rewards = rewards.clone().detach().cpu().numpy()
    cummulative_rewards = [rewards[:index].sum() for index in range(len(rewards))]
    advantage_list = []
    advantage = 0.0

    for delta in rewards[::-1]:
        advantage = gamma * advantage_param * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()

    return torch.tensor(
        np.array(advantage_list), dtype=torch.float, device=device
    )  # [B]


def _process_supervision(
    gamma: float, advantage_param: float, rewards: Tensor, device: torch.cuda.device
):
    """
    ## Arguments:

        gamma (float)

        advantage_param (float)

        rewards (Tensor): [EPISODE, B, AGENT_NUM]

        device (torch.cuda.device)
    """
    pass
