import torch
import torch.nn.functional as F

__all__ = ['focal_loss']


def focal_loss(input: torch.Tensor, target: torch.Tensor, gamma: float = 0.,
               label_smoothing: float = 0.0) -> torch.Tensor:
    logp = F.cross_entropy(input, target, label_smoothing=label_smoothing)
    p = torch.exp(-logp)
    loss = (1 - p) ** gamma * logp
    return loss.mean()
