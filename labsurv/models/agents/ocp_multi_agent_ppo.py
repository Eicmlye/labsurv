import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from labsurv.builders import AGENTS, STRATEGIES
from labsurv.models.agents import BaseAgent
from labsurv.runners.hooks import LoggerHook
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

        super().__init__(device, gamma)

        self.test_mode = test_mode
        self.mixed_reward = mixed_reward
        self.cam_types = cam_types
        self.agent_num = agent_num

        self.actor: Module = STRATEGIES.build(actor_cfg).to(self.device)
        self.critic: Module = STRATEGIES.build(critic_cfg).to(self.device)

        if not self.test_mode:
            self.lr = [actor_lr, critic_lr]
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr[0])
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr[1])

            self.pan_section_num = pan_section_num
            self.tilt_section_num = tilt_section_num
            self.pan_range = pan_range
            self.tilt_range = tilt_range

            self.start_episode = 0
            self.update_step = update_step
            self.advantage_param = advantage_param
            self.clip_epsilon = clip_epsilon
            self.entropy_loss_coef = entropy_loss_coef

        if resume_from is not None:
            self.resume(resume_from)
        elif load_from is not None:
            self.load(load_from)

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

    def take_action(
        self, room, cam_params: array, **kwargs
    ) -> Tuple[List[Tuple[array, array, array]], List[array]]:
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
    ) -> Tuple[List[Tuple[array, array, array]], List[array]]:
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

        with torch.no_grad():  # if grad, memory leaks
            observations: List[Tuple[array, array, array]] = []
            actions: List[array] = []
            action_masks: List[array] = []

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
                print(
                    f"[Episode {kwargs["episode_index"] + 1:>4}  "
                    f"Step {kwargs["step_index"] + 1:>3}] "
                    f"action dist for agent {agent_index + 1} "
                    "\n           x          y          z          p          t",
                    end="",
                )
                action_probs = action_dist.clone().detach()
                action_probs[action_mask == 1] = -1
                action_probs_list = [float(i) for i in action_probs.tolist()]
                print("\n   -   ", end="")
                for i in range(5):
                    print(f"{action_probs_list[2 * i]:>9.6f}  ", end="")
                print("\n   +   ", end="")
                for i in range(5):
                    print(f"{action_probs_list[2 * i + 1]:>9.6f}  ", end="")
                print("\n  cam  ", end="")
                for i in range(len(action_probs_list) - 10):
                    print(f"    {i}     ", end="")
                print("\n       ", end="")
                for i in range(10, len(action_probs_list)):
                    print(f"{action_probs_list[i]:>9.6f}  ", end="")
                print(f"\nEntropy: {action_dist_cat.entropy().item():.6f}")

                import pdb

                pdb.set_trace()
                action_index = action_dist_cat.sample().type(torch.int64).item()

                actions.append(np.eye(len(action_dist))[[action_index]].reshape(-1))
                action_masks.append(action_mask)

        # [AGENT_NUM, Tuple], [AGENT_NUM, ACTION_DIM], [AGENT_NUM, ACTION_DIM]
        return observations, actions, action_masks

    def test_take_action(
        self, room, cam_params: array, **kwargs
    ) -> Tuple[List[Tuple[array, array, array]], List[array]]:
        """
        ## Arguments:

            room (SurveillanceRoom).

            cam_params (np.ndarray): [AGENT_NUM, PARAM_DIM]. Absolute params of agents.

        ## Returns:

            observations (List[Tuple[array, array, array]]): [AGENT_NUM, Tuple], actor
            inputs.

            actions (np.ndarray): [AGENT_NUM, ACTION_DIM].
        """

        with torch.no_grad():  # if grad, memory leaks
            observations: List[Tuple[array, array, array]] = []
            actions: List[array] = []

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
                action_dist: Tensor = F.softmax(
                    self.actor(self_and_neigh_params, self_mask, neigh).view(-1),
                    dim=-1,
                )

                # DEBUG(eric)
                print(
                    f"[Step {kwargs["step_index"] + 1:>3}] "
                    f"action dist for agent {agent_index + 1} "
                    "\n           x          y          z          p          t",
                    end="",
                )
                action_dist_cat = torch.distributions.Categorical(action_dist)
                action_probs = [float(i) for i in action_dist.tolist()]
                print("\n   -   ", end="")
                for i in range(5):
                    print(f"{action_probs[2 * i]:>9.6f}  ", end="")
                print("\n   +   ", end="")
                for i in range(5):
                    print(f"{action_probs[2 * i + 1]:>9.6f}  ", end="")
                print("\n  cam  ", end="")
                for i in range(len(action_probs) - 10):
                    print(f"    {i}     ", end="")
                print("\n       ", end="")
                for i in range(10, len(action_probs)):
                    print(f"{action_probs[i]:>9.6f}  ", end="")
                print(f"\nEntropy: {action_dist_cat.entropy().item():.6f}")

                action_index = action_dist_cat.sample().type(torch.int64).item()

                actions.append(np.eye(len(action_dist))[[action_index]].reshape(-1))

        # [AGENT_NUM, Tuple], [AGENT_NUM, ACTION_DIM]
        return observations, actions

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

        ## permute batch channel and agent channel for actor inputs
        batch_size = len(cur_observations)
        cur_self_and_neigh_params_list: List[List[array]] = [
            [] for i in range(self.agent_num)
        ]
        cur_self_mask_list: List[List[array]] = [[] for i in range(self.agent_num)]
        cur_neigh_list: List[List[array]] = [[] for i in range(self.agent_num)]
        cur_all_actions_list: List[List[array]] = [[] for i in range(self.agent_num)]
        cur_all_action_masks_list: List[List[array]] = [
            [] for i in range(self.agent_num)
        ]
        for batch_index in range(batch_size):
            for agent_index in range(self.agent_num):
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
            device=self.device,
        )
        cur_self_mask: Tensor = torch.tensor(  # [AGENT_NUM, B, AGENT_NUM(NEIGH)]
            np.array(cur_self_mask_list), dtype=torch.bool, device=self.device
        )
        cur_neigh: Tensor = torch.tensor(  # [AGENT_NUM, B, 3, 2L+1, 2L+1, 2L+1]
            np.array(cur_neigh_list), dtype=torch.float, device=self.device
        )
        cur_all_actions: Tensor = torch.tensor(  # [AGENT_NUM, B, ACTION_DIM]
            np.array(cur_all_actions_list), dtype=torch.float, device=self.device
        )
        cur_all_action_masks: Tensor = torch.tensor(  # [AGENT_NUM, B, ACTION_DIM]
            np.array(cur_all_action_masks_list), dtype=torch.bool, device=self.device
        )

        ## critic inputs
        cur_cam_params: Tensor = torch.tensor(  # [B, AGENT_NUM, PARAM_DIM]
            np.array([cur_critic_input[0] for cur_critic_input in cur_critic_inputs]),
            dtype=torch.float,
            device=self.device,
        )
        cur_envs: Tensor = torch.tensor(  # [B, 3, W, D, H]
            np.array([cur_critic_input[1] for cur_critic_input in cur_critic_inputs]),
            dtype=torch.float,
            device=self.device,
        )
        next_cam_params: Tensor = torch.tensor(  # [B, AGENT_NUM, PARAM_DIM]
            np.array(
                [next_critic_input[0] for next_critic_input in next_critic_inputs]
            ),
            dtype=torch.float,
            device=self.device,
        )
        next_envs: Tensor = torch.tensor(  # [B, 3, W, D, H]
            np.array(
                [next_critic_input[1] for next_critic_input in next_critic_inputs]
            ),
            dtype=torch.float,
            device=self.device,
        )
        all_rewards: Tensor = torch.tensor(  # [B, AGENT_NUM + 1]
            rewards, dtype=torch.float, device=self.device
        )
        mixed_rewards: Tensor = all_rewards[:, :-1]  # [B, AGENT_NUM]
        system_rewards: Tensor = all_rewards[:, -1]  # [B]
        all_terminated: Tensor = torch.tensor(  # [B]
            terminated, dtype=self.INT, device=self.device
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
