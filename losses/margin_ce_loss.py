import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import functional as my_F


class MarginCELinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int,
                 dtype=None, device=None):
        super().__init__(in_features, out_features, bias=False, dtype=dtype, device=device)

    @classmethod
    def from_output_layer(cls, output_layer: nn.Linear) -> 'MarginCELinear':
        layer = cls(output_layer.in_features, output_layer.out_features)
        layer.to(dtype=output_layer.weight.dtype, device=output_layer.weight.device)
        layer.weight.data.copy_(output_layer.weight.data)
        layer.requires_grad_(output_layer.weight.requires_grad)
        return layer

    def linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(input), F.normalize(self.weight))


class MarginCELoss(nn.Module):
    r"""Base class for Margin-based Cross Entropy Losses."""

    def __init__(self, output_layer: nn.Linear,
                 eps: float = 1e-7):
        super().__init__()
        self.output_layer = output_layer
        self.patch_output_layer()
        self.eps = eps

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

    def compute_logits(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(input), F.normalize(self.output_layer.weight))

    def compute_cosine(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self.compute_cosine(input, target)
        return self.compute_loss(logits, target)

    def patch_output_layer(self, remove_bias: bool = True) -> None:
        if self.output_layer.bias is not None:
            if remove_bias:
                self.output_layer.register_parameter('bias', None)
            else:
                self.output_layer.bias.data.zero_()

    def patch_model(self, model: nn.Module, output_module_name: str) -> nn.Module:
        output_layer = model.get_submodule(output_module_name)
        assert isinstance(output_layer, nn.Linear)
        model.set_submodule(output_module_name, MarginCELinear.from_output_layer(output_layer))
        return model


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
        super().__init__(output_layer, eps=eps)
        self.margin = margin
        self.phi_flag = phi_flag
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.lambda_min = 5.0
        self.iter = 0

        # duplication formula
        self.mlambda = self.mlambdas[margin]

    def compute_cosine(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.iter += 1
        self.lamb = max(self.lambda_min, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        cos_theta = self.compute_logits(input)
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

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return my_F.focal_loss(logits, target)

    def extra_repr(self) -> str:
        return f'margin={self.margin}, phi_flag={self.phi_flag}'


class CosFace(MarginCELoss):
    r"""
    CosFace Module.
    """

    def __init__(self, output_layer: nn.Linear,
                 scale: float = 64.0, margin: float = 0.40,
                 eps: float = 1e-7):
        super().__init__(output_layer, eps=eps)
        self.scale = scale
        self.margin = margin

    def compute_cosine(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine = self.compute_logits(input)
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
        super().__init__(output_layer, eps=eps)
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def compute_cosine(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine = self.compute_logits(input)
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


class CurricularFace(MarginCELoss):
    def __init__(self, output_layer: nn.Linear,
                 scale: float = 64., margin: float = 0.5,
                 eps: float = 1e-7):
        super().__init__(output_layer, eps=eps)
        self.scale = scale
        self.margin = margin

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.register_buffer('t', torch.zeros(1))

    def compute_cosine(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cos_theta = self.compute_logits(input)
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


class MagFace(MarginCELoss):
    r"""
    MagFace module.
    """

    def __init__(self, out_layer: nn.Linear,
                 scale: float = 64.0,
                 l_a: float = 10, u_a: float = 110, l_margin: float = 0.45, u_margin: float = 0.8,
                 easy_margin: bool = True,
                 lambda_g: float = 20.0,
                 eps: float = 1e-7):
        super().__init__(out_layer, eps=eps)
        self.scale = scale
        self.l_a = l_a
        self.u_a = u_a
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.easy_margin = easy_margin
        self.lambda_g = lambda_g

    def compute_cosine(self, x, m):
        """
        Here m is a function which generate adaptive margin
        """
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        ada_margin = m(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)

        # norm the weight
        weight_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(F.normalize(x), weight_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(
                cos_theta > threshold, cos_theta_m, cos_theta - mm)
        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta

        # regularizer
        return [cos_theta, cos_theta_m], x_norm

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output = self.compute_cosine(input, target)
        g = torch.mean(1 / (self.u_a ** 2) * x_norm + 1 / x_norm)
        return my_F.focal_loss(output, target) + self.lambda_g * g

    def extra_repr(self) -> str:
        return (f'scale={self.scale}, '
                f'a_bound={(self.l_a, self.u_a)}, '
                f'margin_bound={(self.l_margin, self.u_margin)}, '
                f'lambda_g={self.lambda_g}')


class AdaFace(MarginCELoss):
    def __init__(self, output_layer: nn.Linear,
                 scale: float = 64.,
                 margin: float = 0.4,
                 h: float = 0.333,
                 t_alpha: float = 1.0,
                 eps: float = 1e-7):
        super().__init__(output_layer, eps=eps)
        self.scale = scale
        self.margin = margin
        self.h = h

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('batch_mean', torch.tensor(20.))
        self.register_buffer('batch_std', torch.tensor(100.))

    def compute_cosine(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
