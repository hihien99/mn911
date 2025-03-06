import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class Visualizer(nn.Module):
    r"""Visualizable Module Wrapper"""

    def __init__(self, model: nn.Module, output_layer: nn.Module = None, dim: int = 3):
        super().__init__()
        self.dim = dim
        self.model = model
        if output_layer is None:
            for m in model.modules():
                output_layer = m  # get the last layer
        assert isinstance(output_layer, nn.Linear)
        features_dim, num_classes = output_layer.in_features, output_layer.out_features
        dtype, device = output_layer.weight.dtype, output_layer.weight.device
        self.vis_fc = torch.nn.Linear(features_dim, dim, bias=False,
                                      dtype=dtype, device=device)
        self.fc = torch.nn.Linear(dim, num_classes, bias=output_layer.bias is not None,
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
