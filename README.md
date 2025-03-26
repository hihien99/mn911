MN911 - Advanced Biometrics Security Mini Project
------

This repo contains the code of the **MN911 - Mini-project**, in which we train the ResNet architecture
using different margin-based losses.
The final report can be found in [`resources/CNN-project6.pdf`](resources/CNN-project6.pdf).

## Usage

### Training

These commands are used to train the ResNet-18 model.
Modify all `resnet18` to `resnet50` to train with a ResNet-50 model.

- With Cross Entropy Loss:
```cmd
python train.py resnet18 --dataset cifar10 --model resnet18 --manual_seed 1 --batch_size 256 --epochs 50
```
- With Margin-based Cross Entropy Losses
  _(available options for `[loss_name]` are `sphereface | cosface | arcface | curricularface | adaface`)_:
```cmd
python train.py resnet18_[loss_name] --dataset cifar10 --model resnet18 --manual_seed 1 --batch_size 256 --epochs 50 --loss [loss_name]
```
