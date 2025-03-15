import os
import os.path as osp
import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from labsurv.builders import IMITATORS, STRATEGIES
from labsurv.models.imitators import BaseImitator
from labsurv.runners.hooks import LoggerHook
from labsurv.utils.surveillance import reformat_input
from mmcv.utils import ProgressBar
from numpy import ndarray as array
from torch import Tensor
from torch.nn import BCELoss, Module


@IMITATORS.register_module()
class AIRL(BaseImitator):
    INT = torch.int64
    FLOAT = torch.float

    def __init__(
        self,
        reward_approximator_cfg: Dict,
        reward_shaping_cfg: Dict,
        device: Optional[str] = None,
        appr_lr: float = 1e-4,
        shaping_lr: float = 1e-4,
        gradient_accumulation_batchsize: Optional[int] = None,
        agent_num: int = 2,
        load_from: Optional[str] = None,
        resume_from: Optional[str] = None,
        backbone_path: Optional[str] = None,
        freeze_backbone: List[int] = [],
        expert_data_path: Optional[str] = None,
        do_reward_change: bool = True,
        truth_threshold: float = 0.2,
        seed: Optional[int] = None,
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

        super().__init__(device)
        self._random = random.Random(seed)

        self.agent_num = agent_num
        self.start_lr = [appr_lr, shaping_lr]
        self.lr = [appr_lr, shaping_lr]
        self.do_reward_change = do_reward_change
        self.truth_threshold = truth_threshold

        self.reward_approximator: Module = STRATEGIES.build(reward_approximator_cfg).to(
            self.device
        )
        self.reward_shaping: Module = STRATEGIES.build(reward_shaping_cfg).to(
            self.device
        )
        self.gradient_accumulation_batchsize = (
            gradient_accumulation_batchsize
            if gradient_accumulation_batchsize is not None
            else self.agent_num
        )

        if len(freeze_backbone) > 0:
            freeze_name = tuple(
                [f"set_abstraction.{index}" for index in freeze_backbone]
            )
            for name, parameter in self.reward_approximator.backbone.named_parameters():
                if name.startswith(freeze_name):
                    parameter.requires_grad = False
            for name, parameter in self.reward_shaping.backbone.named_parameters():
                if name.startswith(freeze_name):
                    parameter.requires_grad = False

            self.appr_opt = torch.optim.Adam(
                filter(
                    lambda p: p.requires_grad, self.reward_approximator.parameters()
                ),
                lr=self.lr[0],
            )
            self.shaping_opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.reward_shaping.parameters()),
                lr=self.lr[1],
            )
        else:
            self.appr_opt = torch.optim.Adam(
                self.reward_approximator.parameters(), lr=self.lr[0]
            )
            self.shaping_opt = torch.optim.Adam(
                self.reward_shaping.parameters(), lr=self.lr[1]
            )

        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)
        elif backbone_path is not None:
            self.load_backbone(backbone_path)

        self.load_data(expert_data_path)

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        missing_keys, unexpected_keys = self.reward_approximator.load_state_dict(
            checkpoint["reward_approximator"]["model_state_dict"], strict=False
        )
        if len(missing_keys) > 0:
            print(f"Missing keys in reward_approximator checkpoint: \n{missing_keys}")
        if len(unexpected_keys) > 0:
            print(
                f"Unexpected keys in reward_approximator checkpoint: \n{unexpected_keys}"
            )

        missing_keys, unexpected_keys = self.reward_shaping.load_state_dict(
            checkpoint["reward_shaping"]["model_state_dict"], strict=False
        )
        if len(missing_keys) > 0:
            print(f"Missing keys in reward_shaping checkpoint: \n{missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys in reward_shaping checkpoint: \n{unexpected_keys}")
        # One shall not load params of the optimizers, because learning rate
        # is contained in the state_dict of the optimizers, and loading
        # optimizer params will ignore the new learning rate.

    def resume(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)

        missing_keys, unexpected_keys = self.reward_approximator.load_state_dict(
            checkpoint["reward_approximator"]["model_state_dict"], strict=False
        )
        if len(missing_keys) > 0:
            print(f"Missing keys in reward_approximator checkpoint: \n{missing_keys}")
        if len(unexpected_keys) > 0:
            print(
                f"Unexpected keys in reward_approximator checkpoint: \n{unexpected_keys}"
            )
        missing_keys, unexpected_keys = self.reward_shaping.load_state_dict(
            checkpoint["reward_shaping"]["model_state_dict"], strict=False
        )
        if len(missing_keys) > 0:
            print(f"Missing keys in reward_shaping checkpoint: \n{missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys in reward_shaping checkpoint: \n{unexpected_keys}")

        missing_keys, unexpected_keys = self.appr_opt.load_state_dict(
            checkpoint["reward_approximator"]["optimizer_state_dict"], strict=False
        )
        if len(missing_keys) > 0:
            print(
                f"Missing keys in reward_approximator optimizer checkpoint: \n{missing_keys}"
            )
        if len(unexpected_keys) > 0:
            print(
                "Unexpected keys in reward_approximator optimizer checkpoint: "
                f"\n{unexpected_keys}"
            )
        missing_keys, unexpected_keys = self.shaping_opt.load_state_dict(
            checkpoint["reward_shaping"]["optimizer_state_dict"], strict=False
        )
        if len(missing_keys) > 0:
            print(
                f"Missing keys in reward_shaping optimizer checkpoint: \n{missing_keys}"
            )
        if len(unexpected_keys) > 0:
            print(
                "Unexpected keys in reward_shaping optimizer checkpoint: "
                f"\n{unexpected_keys}"
            )

        self.start_episode = checkpoint["episode"] + 1

    def load_backbone(self, backbone_path: str):
        """
        ## Description:

            Load pretrained params for PointNet++ backbone module.
        """
        backbone = torch.load(backbone_path)
        self.reward_approximator.backbone.load_state_dict(backbone)

    def load_data(self, expert_data_path: str):
        self._buffer: List[
            Dict[str, List[bool | float | array | Tuple[int, array]]]
        ] = []

        pkl_list = os.listdir(expert_data_path)
        for data_path in pkl_list:
            with open(osp.join(expert_data_path, data_path), "rb") as f:
                cur_transitions = pickle.load(f)
                for batch_index in range(len(cur_transitions["cur_observation"])):
                    transition = dict()
                    for key, val in cur_transitions.items():
                        transition[key] = val[batch_index]

                    self._buffer.append(transition)

    def sample(self, batch_size: int) -> Dict[str, List[bool | float | array]]:
        batch_transitions: List[Dict[str, bool | float | array]] = self._random.sample(
            self._buffer, batch_size
        )

        samples: Dict[str, List[bool | float | array]] = {
            key: [] for key in batch_transitions[0].keys()
        }
        for transition in batch_transitions:
            for key, val in transition.items():
                samples[key].append(val)

        return samples

    def train(
        self,
        transitions: Dict[str, List[bool | float | array | Tuple[int, array]]],
        **kwargs,
    ):
        logger: LoggerHook = kwargs["logger"]
        actor: Module = kwargs["actor"]
        gamma: float = kwargs["gamma"]

        (
            cur_self_and_neigh_params,  # [AGENT_NUM, B, AGENT_NUM(NEIGH), PARAM_DIM]
            cur_self_mask,  # [AGENT_NUM, B, AGENT_NUM(NEIGH)]
            cur_neigh,  # [AGENT_NUM, B, 3, 2L+1, 2L+1, 2L+1]
            cur_all_actions,  # [AGENT_NUM, B, ACTION_DIM]
            cur_all_action_masks,  # [AGENT_NUM, B, ACTION_DIM]
            next_self_and_neigh_params,  # [AGENT_NUM, B, AGENT_NUM(NEIGH), PARAM_DIM]
            next_self_mask,  # [AGENT_NUM, B, AGENT_NUM(NEIGH)]
            next_neigh,  # [AGENT_NUM, B, 3, 2L+1, 2L+1, 2L+1]
        ) = self._reformat_input(transitions)
        batch_size: int = cur_self_and_neigh_params.shape[1]

        expert_transitions = self.sample(batch_size)
        (
            expert_self_and_neigh_params,  # [AGENT_NUM, B, AGENT_NUM(NEIGH), PARAM_DIM]
            expert_self_mask,  # [AGENT_NUM, B, AGENT_NUM(NEIGH)]
            expert_neigh,  # [AGENT_NUM, B, 3, 2L+1, 2L+1, 2L+1]
            expert_all_actions,  # [AGENT_NUM, B, ACTION_DIM]
            expert_all_action_masks,  # [AGENT_NUM, B, ACTION_DIM]
            expert_next_self_and_neigh_params,  # [AGENT_NUM, B, AGENT_NUM(NEIGH), PARAM_DIM]
            expert_next_self_mask,  # [AGENT_NUM, B, AGENT_NUM(NEIGH)]
            expert_next_neigh,  # [AGENT_NUM, B, 3, 2L+1, 2L+1, 2L+1]
        ) = self._reformat_input(expert_transitions)

        gradient_accumulation_batchnum = int(
            np.ceil(batch_size * self.agent_num / self.gradient_accumulation_batchsize)
        )
        for param_group in self.appr_opt.param_groups:
            param_group["lr"] = self.lr[0] / gradient_accumulation_batchnum
        for param_group in self.shaping_opt.param_groups:
            param_group["lr"] = self.lr[1] / gradient_accumulation_batchnum

        rewards_list: List[float] = []
        agent_probs: List[float] = []
        expert_probs: List[float] = []
        self.appr_opt.zero_grad()
        self.shaping_opt.zero_grad()
        print("\nGradient accumulation for AIRL...")
        prog_bar = ProgressBar(gradient_accumulation_batchnum)
        for ga_step in range(gradient_accumulation_batchnum):
            lower_index = ga_step * self.gradient_accumulation_batchsize
            upper_index_excluded = min(
                (ga_step + 1) * self.gradient_accumulation_batchsize,
                batch_size * self.agent_num,
            )

            agent_disc_prob = _compute_disc_prob(
                lower_index,
                upper_index_excluded,
                cur_self_and_neigh_params,
                cur_self_mask,
                cur_neigh,
                cur_all_actions,
                cur_all_action_masks,
                next_self_and_neigh_params,
                next_self_mask,
                next_neigh,
                self.reward_approximator,
                self.reward_shaping,
                actor,
                gamma,
            )
            expert_disc_prob = _compute_disc_prob(
                lower_index,
                upper_index_excluded,
                expert_self_and_neigh_params,
                expert_self_mask,
                expert_neigh,
                expert_all_actions,
                expert_all_action_masks,
                expert_next_self_and_neigh_params,
                expert_next_self_mask,
                expert_next_neigh,
                self.reward_approximator,
                self.reward_shaping,
                actor,
                gamma,
            )

            discriminator_loss: Tensor = BCELoss()(
                agent_disc_prob, torch.ones_like(agent_disc_prob)
            ) + BCELoss()(expert_disc_prob, torch.zeros_like(expert_disc_prob))

            discriminator_loss.backward()

            rewards_list += (
                torch.log(agent_disc_prob) - torch.log(1 - agent_disc_prob)
            ).tolist()
            agent_probs += agent_disc_prob.view(-1).tolist()
            expert_probs += expert_disc_prob.view(-1).tolist()

            prog_bar.update()
        print("\r\033[K\033[1A\033[K", end="")

        self.appr_opt.step()
        self.shaping_opt.step()

        accuracy, precision, recall = _compute_precision_recall(
            agent_probs, expert_probs, self.truth_threshold
        )

        logger.show_log(
            f"Discriminator acc = {accuracy:.10f} | prec = {precision:.10f} "
            f"| recall = {recall:.10f} | threshold {self.truth_threshold:.8f}",
            with_time=True,
        )

        if self.do_reward_change:
            transitions["reward"] = torch.cat(  # [B, AGENT_NUM + 1]
                (
                    torch.tensor(rewards_list).view(self.agent_num, -1).permute(1, 0),
                    torch.zeros((batch_size, 1)),  # fake system reward
                ),
                dim=1,
            ).tolist()

        return transitions, discriminator_loss.item()

    def _reformat_input(
        self,
        transitions: Dict[str, List[bool | float | array | Tuple[int, array]]],
    ):
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
            next_self_and_neigh_params,  # [AGENT_NUM, B, AGENT_NUM(NEIGH), PARAM_DIM]
            next_self_mask,  # [AGENT_NUM, B, AGENT_NUM(NEIGH)]
            next_neigh,  # [AGENT_NUM, B, 3, 2L+1, 2L+1, 2L+1]
        ) = actor_inputs

        return (
            cur_self_and_neigh_params,  # [AGENT_NUM, B, AGENT_NUM(NEIGH), PARAM_DIM]
            cur_self_mask,  # [AGENT_NUM, B, AGENT_NUM(NEIGH)]
            cur_neigh,  # [AGENT_NUM, B, 3, 2L+1, 2L+1, 2L+1]
            cur_all_actions,  # [AGENT_NUM, B, ACTION_DIM]
            cur_all_action_masks,  # [AGENT_NUM, B, ACTION_DIM]
            next_self_and_neigh_params,  # [AGENT_NUM, B, AGENT_NUM(NEIGH), PARAM_DIM]
            next_self_mask,  # [AGENT_NUM, B, AGENT_NUM(NEIGH)]
            next_neigh,  # [AGENT_NUM, B, 3, 2L+1, 2L+1, 2L+1]
        )

    def update_scheduler(self, cur_episode: int, total_episode: int):
        """
        ## Description:

            Linear one-cycle scheduler.
        """
        cur_episode = min(cur_episode, total_episode - 1)
        for index in range(len(self.lr)):
            self.lr[index] = (
                self.start_lr[index]
                - cur_episode / (total_episode - 1) * (1 - 1e-2) * self.start_lr[index]
            )

    def save(self, episode_index: int, save_path: str):
        checkpoint = dict(
            reward_approximator=dict(
                model_state_dict=self.reward_approximator.state_dict(),
                optimizer_state_dict=self.appr_opt.state_dict(),
            ),
            reward_shaping=dict(
                model_state_dict=self.reward_shaping.state_dict(),
                optimizer_state_dict=self.shaping_opt.state_dict(),
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


def _compute_disc_prob(
    lower_index: int,
    upper_index_excluded: int,
    cur_self_and_neigh_params: Tensor,
    cur_self_mask: Tensor,
    cur_neigh: Tensor,
    cur_all_actions: Tensor,
    cur_all_action_masks: Tensor,
    next_self_and_neigh_params: Tensor,
    next_self_mask: Tensor,
    next_neigh: Tensor,
    reward_approximator: Module,
    reward_shaping: Module,
    actor: Module,
    gamma: float,
) -> Tensor:
    agent_num, batch_size, _, param_dim = cur_self_and_neigh_params.shape
    neigh_side_length: int = cur_neigh.shape[3]

    ga_cur_self_and_neigh_params = cur_self_and_neigh_params.view(
        batch_size * agent_num, -1, param_dim
    )[lower_index:upper_index_excluded]
    ga_cur_self_mask = cur_self_mask.view(batch_size * agent_num, -1)[
        lower_index:upper_index_excluded
    ]
    ga_cur_neigh = cur_neigh.view(
        batch_size * agent_num,
        3,
        neigh_side_length,
        neigh_side_length,
        neigh_side_length,
    )[lower_index:upper_index_excluded]
    ga_cur_all_actions = cur_all_actions.view(batch_size * agent_num, -1)[
        lower_index:upper_index_excluded
    ]
    ga_cur_all_action_masks = cur_all_action_masks.view(batch_size * agent_num, -1)[
        lower_index:upper_index_excluded
    ]
    ga_next_self_and_neigh_params = next_self_and_neigh_params.view(
        batch_size * agent_num, -1, param_dim
    )[lower_index:upper_index_excluded]
    ga_next_self_mask = next_self_mask.view(batch_size * agent_num, -1)[
        lower_index:upper_index_excluded
    ]
    ga_next_neigh = next_neigh.view(
        batch_size * agent_num,
        3,
        neigh_side_length,
        neigh_side_length,
        neigh_side_length,
    )[lower_index:upper_index_excluded]

    rew_appr: Tensor = reward_approximator(
        ga_cur_self_and_neigh_params,
        ga_cur_self_mask,
        ga_cur_neigh,
        ga_cur_all_actions,
    )
    cur_rew_shaping: Tensor = reward_shaping(
        ga_cur_self_and_neigh_params,
        ga_cur_self_mask,
        ga_cur_neigh,
    )
    next_rew_shaping: Tensor = reward_shaping(
        ga_next_self_and_neigh_params,
        ga_next_self_mask,
        ga_next_neigh,
    )

    action_logits: Tensor = actor(
        ga_cur_self_and_neigh_params,
        ga_cur_self_mask,
        ga_cur_neigh,
    )
    action_logits[ga_cur_all_action_masks == 1] = float("-inf")
    action_dist = F.softmax(action_logits, dim=-1)
    action_prob = torch.gather(
        action_dist,
        dim=1,
        index=ga_cur_all_actions.nonzero()[:, -1].view(-1, 1),
    )

    advantage = rew_appr + gamma * next_rew_shaping - cur_rew_shaping
    disc_prob = torch.exp(advantage) / (torch.exp(advantage) + action_prob)

    return disc_prob


def _compute_precision_recall(
    agent_probs: List[float], expert_probs: List[float], threshold: float = 0.2
):
    total_num = len(agent_probs) + len(expert_probs)

    agent_samples: array = np.array(agent_probs)
    expert_samples: array = np.array(expert_probs)
    TP = int((agent_samples > 1 - threshold).sum())
    FP = int((agent_samples <= 1 - threshold).sum())
    TN = int((expert_samples < threshold).sum())
    FN = int((expert_samples >= threshold).sum())

    accuracy = (TP + TN) / total_num
    precision = (TP / (TP + FP)) if TP + FP > 0 else 0.0
    recall = (TP / (TP + FN)) if TP + FN > 0 else 0.0

    return accuracy, precision, recall
