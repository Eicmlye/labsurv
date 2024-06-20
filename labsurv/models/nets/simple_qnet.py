import torch.nn.functional as F
from labsurv.builders import LOSSES, QNETS
from torch.nn import Linear, Module


@QNETS.register_module()
class SimpleQNet(Module):
    def __init__(self, state_dim, hidden_dim, action_dim, loss_cfg):
        super().__init__()
        self.hidden_layer = Linear(state_dim, hidden_dim)
        self.output_layer = Linear(hidden_dim, action_dim)
        self.loss = LOSSES.build(loss_cfg)

    def forward(self, state):
        x = F.relu(self.hidden_layer(state))
        action = self.output_layer(x)

        return action

    def get_loss(self, total_reward, q_target):
        loss = self.loss.get_loss(total_reward, q_target)

        return loss
