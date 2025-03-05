import torch
import torch.nn as nn

__all__ = ['Baseline', 'baseline']


class Baseline(nn.Module):

    def __init__(self,
                 num_classes: int = 10,
                 img_channels: int = 3) -> None:
        super(Baseline, self).__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1024, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=self.num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.relu(x))
        x = self.maxpool(x)

        x = self.conv2(self.relu(x))
        x = self.maxpool(x)

        x = self.conv3(self.relu(x))

        x = self.flatten(x)  # x = x.view(x.size(0), -1)

        x = self.fc1(self.relu(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.fc2(self.softmax(x))
        return x


def baseline(num_classes: int = 10, img_channels: int = 3) -> Baseline:
    r"""Constructs a Baseline model.

    Args:
        num_classes (int): Number of classes.
        img_channels (int): Number of image channels.
    """
    return Baseline(num_classes, img_channels)


if __name__ == '__main__':
    x = torch.rand(8, 3, 32, 32)
    print(x.shape)
    baseline = Baseline(num_classes=10, img_channels=3)
    output = baseline(x)
    print(output.shape)
