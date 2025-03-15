"""
PlainNet architectures modified for low resolution images
"""
from typing import List, Optional, Type, Union, Callable

import torch
import torch.nn as nn

from .resnet import conv3x3, conv1x1

__all__ = ['PlainNet', 'plainnet18', 'plainnet34', 'plainnet50', 'plainnet']


class PlainBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplains: int,
            plains: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None) -> None:
        super(PlainBlock, self).__init__()
        self.conv1 = conv3x3(inplains, plains, stride)
        self.bn1 = nn.BatchNorm2d(plains)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(plains, plains)
        self.bn2 = nn.BatchNorm2d(plains)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class PlainBottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplains: int,
            plains: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            group: int = 1,
            dilation: int = 1,
            base_width: int = 64,
            norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(PlainBottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(plains * (base_width / 64.0)) * group

        self.conv1 = conv1x1(inplains, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, group, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, plains * self.expansion)
        self.bn3 = norm_layer(plains * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        return out


class PlainNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[PlainBlock, PlainBottleneck]],
            layers: List[int],
            num_classes: int = 10,
            img_channels: int = 3,
            zero_init_residual: bool = True,
            low_resolution: bool = True) -> None:
        super(PlainNet, self).__init__()
        self.inplains = 64
        self.img_channels = img_channels
        self.num_classes = num_classes

        if low_resolution:
            # for small datasets: kernel_size 7 -> 3, stride 2 -> 1, padding 3 -> 1
            self.conv1 = nn.Conv2d(self.img_channels, self.inplains, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(self.img_channels, self.inplains, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplains)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        self.reset_parameters(zero_init_residual)

    def reset_parameters(self, zero_init_residual: bool = True) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PlainBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, PlainBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,
                    block: Type[Union[PlainBlock, PlainBottleneck]],
                    plains: int,
                    blocks: int,
                    stride: int = 1):
        downsample = None
        if stride != 1 or self.inplains != plains * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplains, plains * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(plains * block.expansion)
            )

        layers = []
        layers.append(block(self.inplains, plains, stride, downsample))
        self.inplains = plains * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplains, plains))

        return nn.Sequential(*layers)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        logits = self.fc(x)
        return logits


def plainnet18(num_classes: int = 10, img_channels: int = 3, low_resolution: bool = True) -> PlainNet:
    r"""Constructs a PlainNet-18 model (ResNet-18 without skip connections).

    Args:
        num_classes (int): Number of classes.
        img_channels (int): Number of image channels.
        low_resolution (bool): Use low resolution variant or not.
    """
    return PlainNet(PlainBlock, [2, 2, 2, 2], num_classes, img_channels, low_resolution=low_resolution)


def plainnet34(num_classes: int = 10, img_channels: int = 3, low_resolution: bool = True) -> PlainNet:
    r"""Constructs a PlainNet-34 model (ResNet-34 without skip connections).

    Args:
        num_classes (int): Number of classes.
        img_channels (int): Number of image channels.
        low_resolution (bool): Use low resolution variant or not.
    """
    return PlainNet(PlainBlock, [3, 4, 6, 3], num_classes, img_channels, low_resolution=low_resolution)


def plainnet50(num_classes: int = 10, img_channels: int = 3, low_resolution: bool = True) -> PlainNet:
    r"""Constructs a PlainNet-50 model (ResNet-50 without skip connections).

    Args:
        num_classes (int): Number of classes.
        img_channels (int): Number of image channels.
        low_resolution (bool): Use low resolution variant or not.
    """
    return PlainNet(PlainBottleneck, [3, 4, 6, 3], num_classes, img_channels, low_resolution=low_resolution)


def plainnet(depth: int, *args, **kwargs) -> PlainNet:
    r"""Constructs a PlainNet model (ResNet without skip connections)
    given a pre-defined depth.

    Args:
        depth (int): Pre-defined depth.
    """
    construct_func = eval(f'resnet{depth}')
    return construct_func(*args, **kwargs)
