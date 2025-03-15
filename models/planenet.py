"""
PlaneNet architectures modified for low resolution images
"""
from typing import List, Optional, Type, Union, Callable

import torch
import torch.nn as nn

__all__ = ['PlaneNet', 'planenet18', 'planenet34', 'planenet50', 'planenet']


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    r"""
    3x3 convolution with padding
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    r"""
    1x1 convolution
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            group: int = 1,
            dilation: int = 1,
            base_width: int = 64,
            norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * group

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, group, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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


class PlaneNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 10,
            img_channels: int = 3,
            zero_init_residual: bool = True,
            low_resolution: bool = True) -> None:
        super(PlaneNet, self).__init__()
        self.inplanes = 64
        self.img_channels = img_channels
        self.num_classes = num_classes

        if low_resolution:
            # for small datasets: kernel_size 7 -> 3, stride 2 -> 1, padding 3 -> 1
            self.conv1 = nn.Conv2d(self.img_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(self.img_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
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
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int,
                    blocks: int,
                    stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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


def planenet18(num_classes: int = 10, img_channels: int = 3, low_resolution: bool = True) -> PlaneNet:
    r"""Constructs a PlaneNet-18 model (ResNet-18 without skip connections).

    Args:
        num_classes (int): Number of classes.
        img_channels (int): Number of image channels.
        low_resolution (bool): Use low resolution variant or not.
    """
    return PlaneNet(BasicBlock, [2, 2, 2, 2], num_classes, img_channels, low_resolution=low_resolution)


def planenet34(num_classes: int = 10, img_channels: int = 3, low_resolution: bool = True) -> PlaneNet:
    r"""Constructs a PlaneNet-34 model (ResNet-34 without skip connections).

    Args:
        num_classes (int): Number of classes.
        img_channels (int): Number of image channels.
        low_resolution (bool): Use low resolution variant or not.
    """
    return PlaneNet(BasicBlock, [3, 4, 6, 3], num_classes, img_channels, low_resolution=low_resolution)


def planenet50(num_classes: int = 10, img_channels: int = 3, low_resolution: bool = True) -> PlaneNet:
    r"""Constructs a PlaneNet-50 model (ResNet-50 without skip connections).

    Args:
        num_classes (int): Number of classes.
        img_channels (int): Number of image channels.
        low_resolution (bool): Use low resolution variant or not.
    """
    return PlaneNet(Bottleneck, [3, 4, 6, 3], num_classes, img_channels, low_resolution=low_resolution)


def planenet(depth: int, *args, **kwargs) -> PlaneNet:
    r"""Constructs a PlaneNet model (ResNet without skip connections)
    given a pre-defined depth.

    Args:
        depth (int): Pre-defined depth.
    """
    construct_func = eval(f'resnet{depth}')
    return construct_func(*args, **kwargs)
