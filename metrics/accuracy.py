import torch
import torch.nn as nn

__all__ = ['Accuracy', 'AccuracyWithLogits']


class Accuracy(nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return input.eq(target).sum().float().div_(target.numel())


class AccuracyWithLogits(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predictions = torch.argmax(logits, dim=1)
        return predictions.eq(target).sum().float().div_(target.numel())
