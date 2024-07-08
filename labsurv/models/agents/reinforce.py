import numpy as np
import torch
from labsurv.builders import AGENTS, STRATEGIES
from labsurv.models.agents import BaseAgent
from torch import Tensor


@AGENTS.register_module()
class REINFORCE(BaseAgent):
    def __init__(
        self,
        policy_net_cfg,
        device=None,
        gamma=0.9,
        lr=0.1,
    ):
        super().__init__(device, gamma)
        self.policy_net = STRATEGIES.build(policy_net_cfg).to(self.device)
        self.lr = lr

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def take_action(self, observation):
        observation = torch.Tensor(observation).to(self.device)
        actions = self.policy_net(observation)
        action_distribution = torch.distributions.Categorical(probs=actions)

        return action_distribution.sample().item()

    def update(self, markov_chain):
        cur_observations = markov_chain["cur_observation"]
        cur_actions = markov_chain["cur_action"]
        rewards = markov_chain["reward"]

        discounted_reward = Tensor([0]).to(self.device)
        self.optimizer.zero_grad()
        for step in reversed(range(len(cur_observations))):
            cur_observation = Tensor(np.array(cur_observations[step])).to(self.device)
            cur_action = Tensor([cur_actions[step]]).type(torch.int64).to(self.device)
            reward = Tensor([rewards[step]]).to(self.device)

            discounted_reward = self.gamma * discounted_reward + reward
            loss = -discounted_reward * torch.log(
                self.policy_net(cur_observation).gather(0, cur_action)
            )

            loss.backward()
        self.optimizer.step()
