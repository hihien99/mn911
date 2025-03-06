import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import functional as my_F


class MarginCELoss(nn.Module):
    r"""Base class for Margin-based Cross Entropy Losses."""

    def __init__(self, output_layer: nn.Linear):
        super().__init__()
        self.output_layer = output_layer
        self.patch_output_layer()

    @property
    def weight(self) -> torch.Tensor:
        return self.output_layer.weight

    @property
    def in_features(self) -> int:
        return self.output_layer.in_features

    @property
    def out_features(self) -> int:
        return self.output_layer.out_features

    def reset_parameters(self) -> None:
        # nn.init.xavier_uniform_(self.weight)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def compute_logits(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output = self.compute_logits(input, target)
        return F.cross_entropy(output, target)

    def patch_output_layer(self, remove_bias: bool = True) -> None:
        if self.output_layer.bias is not None:
            if remove_bias:
                self.output_layer.register_parameter('bias', None)
            else:
                self.output_layer.bias.data.zero_()

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


def myphi(x, m):
    x = x * m
    return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + \
        x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)


class SphereFace(MarginCELoss):
    r"""
    SphereFace Module.
    """

    mlambdas = [
        lambda x: x ** 0,
        lambda x: x ** 1,
        lambda x: 2 * x ** 2 - 1,
        lambda x: 4 * x ** 3 - 3 * x,
        lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
        lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
    ]

    def __init__(self, output_layer: nn.Linear,
                 margin: int = 4,
                 phi_flag: bool = True,
                 eps: float = 1e-7):
        super().__init__(output_layer)
        self.margin = margin
        self.phi_flag = phi_flag
        self.eps = eps
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.lambda_min = 5.0
        self.iter = 0

        # duplication formula
        self.mlambda = self.mlambdas[margin]

    def compute_logits(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.iter += 1
        self.lamb = max(self.lambda_min, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1 + self.eps, 1 - self.eps)
        if self.phi_flag:
            cos_m_theta = self.mlambda(cos_theta)
            theta = cos_theta.data.acos()
            k = (self.margin * theta / math.pi).floor()
            phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta, self.m)
            phi_theta = phi_theta.clamp(-1 * self.m, 1)

        one_hot = torch.zeros(cos_theta.size(), dtype=input.dtype, device=input.device)
        one_hot.scatter_(1, target.view(-1, 1), 1)

        output = one_hot * (phi_theta - cos_theta) / (1 + self.lamb) + cos_theta
        output.mul_(torch.norm(input, 2, 1).view(-1, 1))

        return output

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output = self.compute_logits(input, target)
        return my_F.focal_loss(output, target)

    def extra_repr(self) -> str:
        return f'margin={self.margin}, phi_flag={self.phi_flag}'


class CosFace(MarginCELoss):
    r"""
    CosFace Module.
    """

    def __init__(self, output_layer: nn.Linear,
                 scale: float = 64.0, margin: float = 0.40,
                 eps: float = 1e-7):
        super().__init__(output_layer)
        self.scale = scale
        self.margin = margin
        self.eps = eps

    def compute_logits(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.margin
        one_hot = torch.zeros(cosine.size(), dtype=torch.bool, device=input.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        # output = one_hot * phi + (1.0 - one_hot) * cosine
        output = torch.where(one_hot, phi, cosine)
        output.mul_(self.scale)
        return output

    def extra_repr(self) -> str:
        return f'scale={self.scale}, margin={self.margin}'


class ArcFace(MarginCELoss):
    r"""
    ArcFace Loss Module.

    ```math
    arcface_loss =-\sum^{m}_{i=1}log
                    \frac{e^{s\psi(\theta_{i,i})}}{e^{s\psi(\theta_{i,i})}+
                    \sum^{n}_{j\neq i}e^{s\cos(\theta_{j,i})}}
    \psi(\theta)=\cos(\theta+m)
    ```
    where $m = margin$, $s = scale$.
    """

    def __init__(self, output_layer: nn.Linear,
                 scale: float = 64.0, margin: float = 0.5, easy_margin: bool = False,
                 eps: float = 1e-7):
        super().__init__(output_layer)
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.eps = eps

    def compute_logits(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1 + self.eps, 1 - self.eps)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(-1 + self.eps, 1 - self.eps))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), dtype=torch.bool, device=input.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        # output = one_hot * phi + (1.0 - one_hot) * cosine
        output = torch.where(one_hot, phi, cosine)
        output.mul_(self.scale)
        return output

    def extra_repr(self) -> str:
        return f'scale={self.scale}, margin={self.margin}, easy_margin={self.easy_margin}'


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

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        index_positive = torch.where(target != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = input > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), input.size(1)], device=input.device)
                mask.scatter_(1, target[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            input = tensor_mul * input

        target_logit = input[index_positive, target[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                input.arccos_()
                final_target_logit = target_logit + self.m2
                input[index_positive, target[index_positive].view(-1)] = final_target_logit
                input.cos_()
            input = input * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            input[index_positive, target[index_positive].view(-1)] = final_target_logit
            input = input * self.s
        else:
            raise

        return input


class CurricularFace(MarginCELoss):
    def __init__(self, output_layer: nn.Linear,
                 scale: float = 64., margin: float = 0.5,
                 eps: float = 1e-7):
        super().__init__(output_layer)
        self.scale = scale
        self.margin = margin
        self.eps = eps

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.register_buffer('t', torch.zeros(1))

    def compute_logits(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1 + self.eps, 1 - self.eps)
        target_logit = cos_theta[torch.arange(0, input.size(0)), target].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, target.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.scale
        return output

    def extra_repr(self) -> str:
        return f'scale={self.scale}, margin={self.margin}'


class AdaFace(MarginCELoss):
    def __init__(self, output_layer: nn.Linear,
                 scale: float = 64.,
                 margin: float = 0.4,
                 h: float = 0.333,
                 t_alpha: float = 1.0,
                 eps: float = 1e-7):
        super().__init__(output_layer)
        self.scale = scale
        self.margin = margin
        self.h = h
        self.eps = eps

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('batch_mean', torch.tensor(20.))
        self.register_buffer('batch_std', torch.tensor(100.))

    def compute_logits(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(input, 2, dim=-1, keepdim=True)
        cosine = F.linear(input / norms, F.normalize(self.weight)).clamp(-1 + self.eps, 1 - self.eps)
        safe_norms = torch.clip(norms, min=self.eps, max=100).clone().detach()  # for stability

        # update statistics
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)  # 66% between -1, 1
        margin_scaler = margin_scaler * self.h  # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(target.size(0), cosine.size(1), dtype=input.dtype, device=input.device)
        m_arc.scatter_(1, target.reshape(-1, 1), 1.0)
        g_angular = -self.margin * margin_scaler
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(target.size(0), cosine.size(1), dtype=input.dtype, device=input.device)
        m_cos.scatter_(1, target.reshape(-1, 1), 1.0)
        g_add = self.margin * (margin_scaler + 1)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        output = cosine * self.scale
        return output

    def extra_repr(self) -> str:
        return f'scale={self.scale}, margin={self.margin}, h={self.h}'
