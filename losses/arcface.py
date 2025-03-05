import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ArcFace', 'CombinedMarginLoss']


class CombinedMarginLoss(nn.Module):
    def __init__(self,
                 s: float = 64.0,
                 m1: float = 1.0,
                 m2: float = 0.5,
                 m3: float = 0.0,
                 interclass_filtering_threshold: float = 0.0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits


class ArcFace(nn.Module):
    r"""
    The input of this Module should be a Tensor which size is (N, embed_size), and the size of output Tensor is (N, num_classes).

    arcface_loss =-\sum^{m}_{i=1}log
                    \frac{e^{s\psi(\theta_{i,i})}}{e^{s\psi(\theta_{i,i})}+
                    \sum^{n}_{j\neq i}e^{s\cos(\theta_{j,i})}}
    \psi(\theta)=\cos(\theta+m)
    where m = margin, s = scale
    """

    def __init__(self, in_features: int, num_classes: int,
                 scale: float = 64.0, margin: float = 0.5, easy_margin: bool = False,
                 bias: bool = False, eps: float = 1e-7):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.in_features = in_features
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(num_classes, in_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_classes, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.eps = eps

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

    def compute_logits(self, embedding: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + self.eps, 1 - self.eps)
        pos = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))
        sin_theta = torch.sqrt((1.0 - torch.pow(pos, 2)).clamp(-1 + self.eps, 1 - self.eps))
        phi = pos * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(pos > 0, phi, pos)
        else:
            phi = torch.where(pos > self.th, phi, pos - self.mm)
        # one_hot = torch.zeros(cos_theta.size(), device='cuda')
        output = torch.scatter(cos_theta, 1, ground_truth.view(-1, 1).long(), phi)
        # output = cos_theta + one_hot
        output.mul_(self.scale)
        return output

    def forward(self, embedding: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        output = self.compute_logits(embedding, ground_truth)
        return F.cross_entropy(output, ground_truth)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, num_classes={self.num_classes}, bias={self.bias is not None},"
                f" scale={self.scale}, margin={self.margin}, easy_margin={self.easy_margin}")

    def patch(self, output_layer: nn.Linear) -> nn.Linear:
        r"""
        Update `nn.Linear` layer with its weights.
        """
        output_layer.weight.data.copy_(self.weight)
        if self.bias is not None:
            if output_layer.bias is not None:
                output_layer.bias.data.copy_(self.bias)
            else:
                raise RuntimeError(f'bias is None')
        elif output_layer.bias is not None:
            output_layer.bias.data.fill_(0)
        return output_layer
