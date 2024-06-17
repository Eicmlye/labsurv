from labsurv.builders import AGENTS, QNETS
import numpy as np
import torch
from torch import Tensor


@AGENTS.register_module()
class DQN:
    def __init__(
        self,
        qnet_cfg,
        lr=0.1,
        gamma=0.9,
        greedy_epsilon=0.2,
        to_target_net_interval=5,
        device=None,
        dqn_type="DQN",
    ):
        self.device = device
        self.qnet = QNETS.build(qnet_cfg).to(self.device)
        self.target_net = QNETS.build(qnet_cfg).to(self.device)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = greedy_epsilon
        self.to_target_net_interval = to_target_net_interval

        self.update_count = 0
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

        assert dqn_type in ["DQN", "DoubleDQN"], f"{dqn_type} not implemented."
        self.dqn_type = dqn_type

    def take_action(self, state):
        if np.random.random() < self.epsilon:  # explore
            return np.random.randint(self.qnet.output_layer.out_features)
        else:  # exploit
            state = torch.from_numpy(state).to(self.device)
            action = self.qnet(state).argmax().item()

            return action

    def update(self, **transition):
        missing_keys = set(
            [
                "states",
                "actions",
                "rewards",
                "next_states",
                "terminated",
                "truncated",
            ]
        ).difference(set(transition.keys()))
        assert len(missing_keys) == 0, f"Missing keys {missing_keys}."

        states = Tensor(np.array(transition["states"])).to(self.device)
        actions = (
            Tensor(transition["actions"]).to(self.device).view(-1, 1).type(torch.int64)
        )
        rewards = Tensor(transition["rewards"]).to(self.device).view(-1, 1)
        next_states = Tensor(np.array(transition["next_states"])).to(self.device)
        terminated = Tensor(transition["terminated"]).to(self.device).view(-1, 1)
        truncated = Tensor(transition["truncated"]).to(self.device).view(-1, 1)

        total_rewards = self.qnet(states).gather(dim=1, index=actions)
        if self.dqn_type == "DQN":
            max_next_total_rewards = (
                self.target_net(next_states).max(dim=1)[0].view(-1, 1)
            )
        elif self.dqn_type == "DoubleDQN":
            max_action = self.qnet(next_states).max(1)[1].view(-1, 1).type(torch.int64)
            max_next_total_rewards = self.target_net(next_states).gather(1, max_action)

        q_targets = rewards + self.gamma * max_next_total_rewards * (1 - terminated) * (
            1 - truncated
        )

        self.optimizer.zero_grad()
        loss = self.qnet.get_loss(total_rewards, q_targets)
        loss.backward()
        self.optimizer.step()

        if self.update_count % self.to_target_net_interval == 0:
            self.target_net.load_state_dict(self.qnet.state_dict())
        self.update_count += 1
