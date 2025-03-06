from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# --------
# Argument
# --------
class tuple_inst:
    def __init__(self, elem_type: type = str, delimiter: str = ','):
        self.elem_type = elem_type
        self.delimiter = delimiter

    def __call__(self, vs):
        if isinstance(vs, str):
            if vs.startswith('[') and vs.endswith(']'):
                vs = vs[1:-1]
            elif vs.startswith('(') and vs.endswith(')'):
                vs = vs[1:-1]
            elif vs.startswith('{') and vs.endswith('}'):
                vs = vs[1:-1]
            return tuple(self.elem_type(v) for v in vs.replace(' ', '').split(self.delimiter))
        else:
            return tuple(self.elem_type(v) for v in vs)


# -------------
# Visualization
# -------------
class Visualizer(nn.Module):
    r"""Visualizable Module Wrapper"""

    def __init__(self, model: nn.Module, output_layer: nn.Module = None, features: int = 3):
        super().__init__()
        self.features = features
        self.model = model
        if output_layer is None:
            for m in model.modules():
                output_layer = m  # get the last layer
        assert isinstance(output_layer, nn.Linear)
        features_dim, num_classes = output_layer.in_features, output_layer.out_features
        dtype, device = output_layer.weight.dtype, output_layer.weight.device
        self.vis_fc = torch.nn.Linear(features_dim, features, bias=False,
                                      dtype=dtype, device=device)
        self.fc = torch.nn.Linear(features, num_classes, bias=output_layer.bias is not None,
                                  dtype=dtype, device=device)
        # copy parameters
        self.vis_fc.weight.data.copy_(torch.linalg.lstsq(self.fc.weight.data, output_layer.weight.data)[0])
        self.fc.bias.data.copy_(output_layer.bias.data)

    @classmethod
    def wrap(cls, model: nn.Module, output_layer: nn.Module = None) -> 'Visualizer':
        return cls(model, output_layer)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.extract_features(x)
        return self.vis_fc(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        return self.fc(x)


def draw_circle(ax: plt.Axes, radius: float, **kwargs):
    return ax.add_patch(plt.Circle((0, 0), radius,
                                   **{'color': (0, 0, 0, 0.2), 'linewidth': 0.2, 'fill': False, **kwargs}))


def draw_globe(ax: plt.Axes, radius: float = 1, resolution: int = 100, **kwargs):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return ax.plot_wireframe(x, y, z,
                             **{'color': (0, 0, 0, 0.2), 'linewidth': 0.2, **kwargs})


def plot_curve(train_acc, val_acc, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(train_acc, label='train_acc')
    ax.plot(val_acc, label='val_acc')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.legend()
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# --------------
# Metric Logging
# --------------

class SmoothedValue(object):
    r"""Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{median:.6f} ({global_avg:.6f})'
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter=', '):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                if v.numel() == 0:
                    v = v.item()
                else:
                    v = v.detach().cpu().numpy()
            assert isinstance(v, (int, float, np.ndarray))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError('"{}" object has no attribute "{}"'.format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                '{}: {}'.format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter
