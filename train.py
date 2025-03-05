"""
Train script for ResNet model on CIFAR10 dataset.

Usage:
    python train.py resnet18 --dataset cifar10 --model resnet18 --batch_size 256 --epochs 50 --use_arcface --device cuda:0
"""

import argparse
import os
import random

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models
import optim
from losses import *
from utils import tuple_inst


def parse_args():
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(__file__)

    parser.add_argument('exp', type=str, nargs='?', default=None,
                        help='experiment name')
    parser.add_argument('--dataset', required=True, default='cifar10',
                        help='dataset to train on: cifar10 | mnist')
    parser.add_argument('--dataroot', default=os.path.join(project_root, 'data'),
                        help='path to dataset')
    parser.add_argument('--data_mean', type=tuple_inst(float), default=[0.4914, 0.4822, 0.4465])
    parser.add_argument('--data_std', type=tuple_inst(float), default=[0.2471, 0.2435, 0.2616])
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size')
    parser.add_argument('--img_channels', type=int, default=3,
                        help='Number of image channels.')
    parser.add_argument('--img_size', type=int, default=32,
                        help='the height / width of the input image to network')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classes in dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--model', required=True,
                        help='baseline | resnet18 | resnet34 | resnet50')
    parser.add_argument('--weights', default='',
                        help='path to pre-trained weights (to continue training)')
    parser.add_argument('--use_arcface', action='store_true',
                        help='use ArcFace loss')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, default=0.001')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training')
    parser.add_argument('--output_dir', default='outputs',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency (epochs)')
    parser.add_argument('--eval_freq', type=int, default=5,
                        help='save frequency (epochs)')
    parser.add_argument('--manual_seed', type=int, default=None,
                        help='manual seed')

    params = parser.parse_args()
    print(params)
    return params


def main():
    params = parse_args()
    params.device = torch.device(params.device)
    os.makedirs(params.output_dir, exist_ok=True)

    if params.manual_seed is not None:
        print('Random Seed: ', params.manual_seed)
        random.seed(params.manual_seed)
        torch.manual_seed(params.manual_seed)
        if params.device.type.startswith('cuda'):
            torch.cuda.manual_seed(params.manual_seed)

    # dataset & data loader
    dataset_cls = getattr(datasets, params.dataset.upper())
    train_dataset = dataset_cls(
        root=params.dataroot, download=True,
        train=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
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
    print('Train set:', train_dataset)
    print('Test set:', test_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.workers)

    # model
    try:
        model = getattr(models, params.model)(
            img_channels=params.img_channels, num_classes=params.num_classes).to(params.device)
    except:
        raise ValueError(f'Model {params.model} is not defined')

    if params.weights != '':
        model.load_state_dict(
            torch.load(params.weights, weights_only=False, map_location=params.device), strict=False)

    # loss
    if params.use_arcface:
        clf_layer = None
        for m in model.modules():
            clf_layer = m
        criterion = ArcFace(clf_layer.in_features, params.num_classes).to(params.device)
        del clf_layer
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr)
    lr_scheduler = optim.lr_scheduler.WarmupCosineLR(optimizer, int(params.epochs * 0.2), params.epochs)

    # create output directory
    output_dir = params.output_dir
    if params.exp is not None:
        output_dir = os.path.join(params.output_dir, params.exp)
    os.makedirs(output_dir, exist_ok=True)

    # training loop
    for epoch in range(1, params.epochs + 1):
        train_loop(train_loader, model, criterion, optimizer, epoch, params)
        if params.use_arcface:
            criterion.patch(model.get_submodule('fc2' if params.model == 'baseline' else 'fc'))
        if lr_scheduler is not None:
            lr_scheduler.step()
        if epoch % params.save_freq == 0:
            torch.save(model.state_dict(), f'{output_dir}/{params.model}_{epoch:03d}.pth')
        if epoch % params.eval_freq == 0:
            print(f'> [Epoch: {epoch:03d}/{params.epochs:03d}]'
                  f' test_acc={eval_loop(model, test_loader, params):.3f}')


def train_loop(train_loader, model, criterion, optimizer, epoch, params):
    model.train()
    for batch_idx, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        X = X.to(params.device)
        y = y.to(params.device)

        if params.use_arcface:
            features = model.extract_features(X)
            loss = criterion(features, y)
        else:
            logits = model(X)
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'- [Epoch {epoch:03d}/{params.epochs:03d} - {batch_idx:04d}/{len(train_loader):04d}]'
                  f' loss={loss:.6f}')


@torch.no_grad()
def eval_loop(model, data_loader, params):
    model.eval()
    correct_pred = 0
    for batch_idx, (X, y) in enumerate(data_loader):
        X = X.to(params.device)
        y = y.to(params.device)

        logits = model(X)
        preds = logits.argmax(1)
        correct_pred += torch.sum(preds.eq(y)).item()
    return correct_pred.float() / len(data_loader.dataset) * 100


if __name__ == '__main__':
    main()
