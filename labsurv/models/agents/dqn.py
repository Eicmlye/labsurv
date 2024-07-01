import numpy as np
import torch
from labsurv.builders import AGENTS, QNETS
from labsurv.models.agents import BaseAgent
from torch import Tensor


@AGENTS.register_module()
class DQN(BaseAgent):
    def __init__(
        self,
        qnet_cfg,
        device=None,
        gamma=0.9,
        explorer_cfg=None,
        lr=0.1,
        to_target_net_interval=5,
        dqn_type="DQN",
    ):
        super().__init__(device, gamma, explorer_cfg)
        self.qnet = QNETS.build(qnet_cfg).to(self.device)
        self.target_net = QNETS.build(qnet_cfg).to(self.device)
        self.lr = lr
        self.to_target_net_interval = to_target_net_interval

        self.update_count = 0
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

        assert dqn_type in ["DQN", "DoubleDQN"], f"{dqn_type} not implemented."
        self.dqn_type = dqn_type

    def take_action(self, observation):
        if self.explorer.decide():
            return np.random.randint(self.qnet.output_layer.out_features)
        else:
            observation = torch.Tensor(observation).to(self.device)
            action = self.qnet(observation).argmax().item()

            return action

    def update(self, samples):
        cur_observations = Tensor(np.array(samples["cur_observation"])).to(self.device)
        cur_actions = (
            Tensor(samples["cur_action"]).to(self.device).view(-1, 1).type(torch.int64)
        )
        rewards = Tensor(samples["reward"]).to(self.device).view(-1, 1)
        next_observations = Tensor(np.array(samples["next_observation"])).to(
            self.device
        )
        terminated = Tensor(samples["terminated"]).to(self.device).view(-1, 1)

        total_rewards = self.qnet(cur_observations).gather(dim=1, index=cur_actions)
        if self.dqn_type == "DQN":
            max_next_total_rewards = (
                self.target_net(next_observations).max(dim=1)[0].view(-1, 1)
            )
        elif self.dqn_type == "DoubleDQN":
            max_action = (
                self.qnet(next_observations).max(1)[1].view(-1, 1).type(torch.int64)
            )
            max_next_total_rewards = self.target_net(next_observations).gather(
                1, max_action
            )

        q_targets = rewards + self.gamma * max_next_total_rewards * (1 - terminated)

        self.optimizer.zero_grad()
        loss = self.qnet.get_loss(total_rewards, q_targets)
        loss.backward()
        self.optimizer.step()

        if self.update_count % self.to_target_net_interval == 0:
            self.target_net.load_state_dict(self.qnet.state_dict())
        self.update_count += 1
