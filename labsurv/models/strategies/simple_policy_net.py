import torch.nn.functional as F
from labsurv.builders import STRATEGIES
from torch.nn import Linear, Module


@STRATEGIES.register_module()
class SimplePolicyNet(Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.conv1 = Linear(state_dim, hidden_dim)
        self.conv2 = Linear(hidden_dim, action_dim)

    def forward(self, observation):
        x = F.relu(self.conv1(observation))
        actions = F.softmax(self.conv2(x), dim=0)

        return actions
