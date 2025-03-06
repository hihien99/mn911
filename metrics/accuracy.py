import torch
import torch.nn as nn

__all__ = ['Accuracy', 'AccuracyWithLogits']


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('corrects', torch.zeros(1))
        self.register_buffer('total', torch.zeros(1))

    def reset_buffers(self):
        self.corrects.zero_()
        self.total.zero_()

    def summarize(self) -> torch.Tensor:
        return self.corrects / self.total

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        corrects = input.eq(target).sum().float()
        total = target.numel()
        batch_accuracy = corrects / total
        self.corrects += corrects
        self.total += total
        return batch_accuracy


class AccuracyWithLogits(Accuracy):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        preds = torch.argmax(logits, dim=1)
        corrects = preds.eq(target).sum().float()
        total = target.numel()
        batch_accuracy = corrects / total
        self.corrects += corrects
        self.total += total
        return batch_accuracy
