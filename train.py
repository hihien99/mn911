"""
Train script for ResNet18 model on Cifar10 dataset.

Usage:
    python train.py --dataset cifar10 --model resnet18 --data_mean [0.485,0.456,0.406] --data_std [0.229,0.224,0.225] --img_size 32 --epochs 35 --img_channels 3 --device cuda:0
"""

import argparse
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models import *
from utils.utils import tuple_inst


def parse_args():
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    parser.add_argument('--dataset', required=True,
                        help='cifar10 | mnist')
    parser.add_argument('--dataroot', default=os.path.join(project_root, 'data'),
                        help='path to dataset')
    parser.add_argument('--data_mean', type=tuple_inst(float), default=None)
    parser.add_argument('--data_std', type=tuple_inst(float), default=None)
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size')
    parser.add_argument('--img_channels', type=int, default=1,
                        help='Number of image channels.')
    parser.add_argument('--img_size', type=int, default=28,
                        help='the height / width of the input image to network')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classes in dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--model', required=True,
                        help='baseline | resnet18 | resnet34 | resnet50')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, default=0.001')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training')
    parser.add_argument('--net', default='',
                        help='path to net (to continue training)')
    parser.add_argument('--outf', default='outputs',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency (epochs)')
    parser.add_argument('--manual_seed', type=int, default=None,
                        help='manual seed')

    params = parser.parse_args()
    print(params)
    return params


def main():
    params = parse_args()
    device = torch.device(params.device)
    os.makedirs(params.outf, exist_ok=True)

    if params.manual_seed is not None:
        print('Random Seed: ', params.manual_seed)
        random.seed(params.manual_seed)
        torch.manual_seed(params.manual_seed)
        if device.type.startswith('cuda'):
            torch.cuda.manual_seed(params.manual_seed)

    # dataset & data loader
    dataset_cls = getattr(datasets, params.dataset.upper())
    train_dataset = dataset_cls(
        root=params.dataroot, download=True,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(params.data_mean, params.data_std),
        ])
    )
    test_dataset = dataset_cls(
        root=params.dataroot, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(params.data_mean, params.data_std),
        ])
    )
    assert train_dataset, test_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=int(params.workers))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=int(params.workers))

    # model
    if params.model == 'baseline':
        model = Baseline(img_channels=params.img_channels,
                         num_classes=params.num_classes).to(device)
    elif params.model == 'resnet18':
        model = resnet18(num_classes=params.num_classes,
                         img_channels=params.img_channels).to(device)
    elif params.model == 'resnet34':
        model = resnet34(num_classes=params.num_classes,
                         img_channels=params.img_channels).to(device)
    else:
        model = resnet50(num_classes=params.num_classes,
                         img_channels=params.img_channels).to(device)
    
    if params.net != '':
        model.load_state_dict(torch.load(params.net))

    # loss
    criterion = nn.CrossEntropyLoss()
    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    os.makedirs(f'{params.outf}/{params.dataset}/{params.model}', exist_ok=True)
    # train loop
    for epoch in range(params.epochs):
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            if not batch_idx % 100:
                print(f'[Epoch {epoch + 1:03d}/{params.epochs:03d} - {batch_idx:04d}/{len(train_loader):04d}]'
                      f' loss={loss:.4f}')

        model.eval()
        print(f'[Epoch: {epoch + 1:03d}/{params.epochs:03d}]'
              f' test_acc={eval_loop(model, test_loader, device=device):.3f}')

        # do checkpointing
        torch.save(model.state_dict(), f'{params.outf}/{params.dataset}/{params.model}/{params.model}_{epoch:03d}.pth')


@torch.no_grad()
def eval_loop(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        probs = model(features).softmax(1)
        predicted_labels = probs.argmax(1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


if __name__ == '__main__':
    main()