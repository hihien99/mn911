import torch
import torch.nn as nn
from . import functional as F

__all__ = ['FocalLoss']


class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.focal_loss(input, target, self.gamma)
