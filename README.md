MN911 - Advanced Biometrics Security Mini Project
------

## Usage

These commands are used to train the ResNet18 model:

- With CrossEntropy Loss:
```cmd
python train.py resnet18 --dataset cifar10 --model resnet18 --manual_seed 1 --batch_size 256 --epochs 50 --device cuda:0
```
- With ArcFace Loss:
```cmd
python train.py resnet18_arcface --dataset cifar10 --model resnet18 --manual_seed 1 --batch_size 256 --epochs 50 --use_arcface --device cuda:0
```

### References

#### Pretrained weights:
- https://github.com/huyvnphan/PyTorch_CIFAR10/
