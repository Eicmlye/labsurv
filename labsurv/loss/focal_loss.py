import torch
from torch.nn import Module
import torch.nn.functional as F

from labsurv.builders import LOSSES


class BinaryFocalLoss(Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, p, targets):
        p = torch.clamp(p, self.eps, 1 - self.eps)
        
        ce_loss = F.binary_cross_entropy(p, targets, reduction='none')
        
        pt = torch.where(targets == 1, p, 1 - p)
        
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss