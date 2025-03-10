"""
Train script for ResNet model on CIFAR10 dataset.

Usage:
    python train.py resnet18 --dataset cifar10 --model resnet18 --batch_size 256 --epochs 100 --device cuda:0
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torcheval.metrics as torch_metrics
import models
import optim
import utils
from losses import *


def parse_args():
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(__file__)

    parser.add_argument('exp', type=str, nargs='?', default=None,
                        help='experiment name')
    parser.add_argument('--dataset', required=True, default='cifar10',
                        help='dataset to train on: cifar10 | mnist')
    parser.add_argument('--dataroot', default=os.path.join(project_root, 'data'),
                        help='path to dataset')
    parser.add_argument('--data_mean', type=utils.tuple_inst(float), default=[0.4914, 0.4822, 0.4465])
    parser.add_argument('--data_std', type=utils.tuple_inst(float), default=[0.2471, 0.2435, 0.2616])
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
    parser.add_argument('--ce_pretrain_epochs', type=int, default=0,
                        help='number of epochs to pretrain with CrossEntropyLoss')
    parser.add_argument('--model', required=True,
                        help='baseline | resnet18 | resnet34 | resnet50')
    parser.add_argument('--weights', default='',
                        help='path to pre-trained weights (to continue training)')
    parser.add_argument('--loss',
                        choices=['ce', 'focal', 'sphereface', 'cosface', 'arcface', 'curricularface', 'adaface'],
                        default='ce',
                        help='loss function to be used')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, default=0.001')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training')
    parser.add_argument('--output_dir', default='outputs',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--ckpt_freq', type=int, default=0,
                        help='save frequency (epochs)')
    parser.add_argument('--eval_freq', type=int, default=5,
                        help='save frequency (epochs)')
    parser.add_argument('--eval_only', action='store_true',
                        help='run only evaluation')
    parser.add_argument('--manual_seed', type=int, default=None,
                        help='manual seed')

    parser.add_argument('--vis', '--visualize', action='store_true',
                        help='train only for visualization')
    parser.add_argument('--vis_dim', type=int, choices=[2, 3], default=3,
                        help='visualization dimensionality')
    parser.add_argument('--vis_freq', type=int, default=5,
                        help='visualization frequency')

    params = parser.parse_args()
    params.use_margin_loss = params.loss not in ['ce', 'focal']
    print(params)
    return params


def main():
    params = parse_args()

    # create output directory
    output_dir = params.output_dir
    if params.exp is not None:
        output_dir = os.path.join(params.output_dir, params.exp)
    if params.eval_only:
        output_dir = os.path.join(output_dir, 'eval')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(vars(params), f)
    params.device = torch.device(params.device)

    # set seed
    if params.manual_seed is not None:
        print('Random Seed: ', params.manual_seed)
        random.seed(params.manual_seed)
        np.random.seed(params.manual_seed)
        torch.manual_seed(params.manual_seed)
        if params.device.type.startswith('cuda'):
            torch.cuda.manual_seed(params.manual_seed)

    # dataset & data loader
    dataset_cls = getattr(datasets, params.dataset.upper())
    train_dataset = dataset_cls(
        root=params.dataroot, download=True,
        train=True,
        transform=transforms.Compose([
            # transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.8, 1.2),
                                   contrast=(0.8, 1.2),
                                   saturation=(0.8, 1.2),
                                   hue=(-0.1, 0.1)),
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

    # visualization wrapper
    if params.vis:
        model = utils.Visualizer(model, features=params.vis_dim).to(params.device)
        img_output_dir = os.path.join(output_dir, 'imgs')
        os.makedirs(img_output_dir, exist_ok=True)

    # loss
    ce_loss = torch.nn.CrossEntropyLoss()
    if params.loss == 'ce':
        criterion = ce_loss
    elif params.loss == 'focal':
        criterion = FocalLoss()
    else:
        clf_layer_name = 'fc2' if params.model == 'baseline' else 'fc'
        clf_layer: nn.Linear = model.get_submodule(clf_layer_name)
        if params.loss == 'sphereface':
            criterion = SphereFace(clf_layer).to(params.device)
        elif params.loss == 'cosface':
            criterion = CosFace(clf_layer).to(params.device)
        elif params.loss == 'arcface':
            criterion = ArcFace(clf_layer).to(params.device)
        elif params.loss == 'curricularface':
            criterion = CurricularFace(clf_layer).to(params.device)
        elif params.loss == 'adaface':
            criterion = AdaFace(clf_layer).to(params.device)
        else:
            criterion = None
        del clf_layer
        criterion.reset_parameters()
        model = criterion.patch_model(model, clf_layer_name)
    print('Loss function:', criterion)

    if len(params.weights):
        model.load_state_dict(
            torch.load(params.weights, weights_only=False, map_location=params.device), strict=False)
        print(f'Loaded weights from {params.weights}')
    elif params.eval_only:
        raise ValueError('No weights provided for evaluation')
    print(model)

    # optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr)
    # scheduler
    lr_scheduler = optim.lr_scheduler.WarmupCosineLR(optimizer, int(params.epochs * 0.3), params.epochs)

    # metrics
    metrics = {
        'accuracy': torch_metrics.MulticlassAccuracy(),
        'precision': torch_metrics.MulticlassPrecision(),
        'recall': torch_metrics.MulticlassRecall(),
        'f1': torch_metrics.MulticlassF1Score(),
        'auroc': torch_metrics.MulticlassAUROC(num_classes=params.num_classes),
        'auprc': torch_metrics.MulticlassAUPRC(num_classes=params.num_classes),
        'confusion_matrix': torch_metrics.MulticlassConfusionMatrix(num_classes=params.num_classes),
    }

    # training loop
    log_file_path = os.path.join(output_dir, 'log.txt')
    if params.eval_only:
        log_stats = {
            'epoch': 'eval',
        }
        eval_stats = eval_loop(model, test_loader, criterion, metrics, None, params)
        log_stats = {**log_stats, **{f'val_{k}': v for k, v in eval_stats.items()}}
        # visualize
        if params.vis:
            fig = visualize_loop(model, train_loader, params)
            fig.savefig(os.path.join(img_output_dir, f'eval.png'))
            plt.close(fig)
        # log
        for k, v in log_stats.items():
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                if torch.is_tensor(v):
                    v = v.cpu()
                csv_path = os.path.join(output_dir, f'{k}.csv')
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                np.savetxt(csv_path, np.asarray(v), delimiter=',')
                log_stats[k] = csv_path
        with open(log_file_path, 'a') as log_f:
            log_f.write(json.dumps(log_stats) + '\n')
    else:
        for epoch in range(1, params.epochs + 1):
            is_eval_epoch = epoch % params.eval_freq == 0 or epoch in [1, params.epochs]
            train_stats = train_loop(
                model, train_loader, criterion if epoch > params.ce_pretrain_epochs else ce_loss,
                optimizer, metrics if is_eval_epoch else {}, epoch, params)
            log_stats = {
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                **{f'train_{k}': v for k, v in train_stats.items()}
            }
            if lr_scheduler is not None:
                lr_scheduler.step()
            # eval
            if is_eval_epoch:
                eval_stats = eval_loop(model, test_loader, criterion, metrics, epoch, params)
                log_stats = {**log_stats, **{f'val_{k}': v for k, v in eval_stats.items()}}
            # log
            for k, v in log_stats.items():
                if torch.is_tensor(v) or isinstance(v, np.ndarray):
                    if torch.is_tensor(v):
                        v = v.cpu()
                    csv_path = os.path.join(output_dir, str(k), f'{epoch:03d}.csv')
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    np.savetxt(csv_path, np.asarray(v), delimiter=',')
                    log_stats[k] = csv_path
            with open(log_file_path, 'a') as log_f:
                log_f.write(json.dumps(log_stats) + '\n')
            # checkpointing
            if (params.ckpt_freq and epoch % params.ckpt_freq == 0) or epoch == params.epochs:
                torch.save(model.state_dict(), os.path.join(output_dir, f'{params.model}_{epoch:03d}.pth'))
            # visualize
            if params.vis and epoch % params.vis_freq == 0:
                fig = visualize_loop(model, train_loader, params)
                fig.savefig(os.path.join(img_output_dir, f'{epoch:03d}.png'))
                plt.close(fig)


def itemize(x):
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        else:
            return x.detach().cpu()
    elif isinstance(x, np.ndarray) and  x.size == 1:
        return x.item()
    return x


def train_loop(model, train_loader, criterion, optimizer, metrics, epoch, params):
    model.train()
    for metric in metrics.values():
        metric.reset()
    metric_logger = utils.MetricLogger()
    for batch_idx, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        X = X.to(params.device)
        y = y.to(params.device)

        if isinstance(criterion, MarginCELoss):
            features = model.extract_features(X)
            loss = criterion(features, y)
            logits = criterion.compute_logits(features)
        else:
            logits = model(X)
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # compute stats
        preds = logits.argmax(1).detach()
        for k, metric in metrics.items():
            metric.update((preds if not k.startswith('au') else logits.softmax(1)).cpu(), y.cpu())
        log_stats = {
            'loss': loss.item(),
            'batch_accuracy': torch_metrics.functional.multiclass_accuracy(preds, y).item(),
        }
        metric_logger.update(**log_stats)

        if batch_idx % 10 == 0:
            print(f'- [Epoch {epoch:03d}/{params.epochs:03d} - {batch_idx:04d}/{len(train_loader):04d}]'
                  f' loss={loss:.6f}')

    print(f'✔️ [Epoch {epoch:03d}/{params.epochs:03d}]', metric_logger)
    stats = {
        **{k: itemize(metric.global_avg) for k, metric in metric_logger.meters.items()},
        **{k: itemize(metric.compute()) for k, metric in metrics.items()}
    }
    for metric in metrics.values():
        metric.reset()
    return stats


@torch.no_grad()
def eval_loop(model, data_loader, criterion, metrics, epoch, params):
    model.eval()
    for metric in metrics.values():
        metric.reset()
    metric_logger = utils.MetricLogger()
    for batch_idx, (X, y) in enumerate(data_loader):
        X = X.to(params.device)
        y = y.to(params.device)

        if isinstance(criterion, MarginCELoss):
            features = model.extract_features(X)
            loss = criterion(features, y)
            logits = criterion.compute_logits(features)
        else:
            logits = model(X)
            loss = criterion(logits, y)

        # compute stats
        preds = logits.argmax(1).detach()
        for k, metric in metrics.items():
            metric.update((preds if not k.startswith('au') else logits.softmax(1)).cpu(), y.cpu())
        log_stats = {
            'loss': loss.item(),
            'batch_accuracy': torch_metrics.functional.multiclass_accuracy(preds, y).item(),
        }
        metric_logger.update(**log_stats)

    if epoch:
        print(f'⭕ [Epoch {epoch:03d}/{params.epochs:03d}]', metric_logger)
    else:
        print('⭕ [Eval]', metric_logger)
    stats = {
        **{k: itemize(metric.global_avg) for k, metric in metric_logger.meters.items()},
        **{k: itemize(metric.compute()) for k, metric in metrics.items()}
    }
    for metric in metrics.values():
        metric.reset()
    return stats


@torch.no_grad()
def visualize_loop(model, train_loader, params):
    model.eval()
    feats = []
    labels = []
    num_samples = 0
    num_samples_max = 50 * params.num_classes
    for batch_idx, (X, y) in enumerate(train_loader):
        if num_samples + X.size(0) + num_samples > num_samples_max:
            X = X[:num_samples_max - num_samples]
            y = y[:num_samples_max - num_samples]
        X = X.to(params.device)
        y = y.to(params.device)

        features = model.extract_features(X)
        feats.append(features.cpu().numpy())
        labels.append(y.cpu().numpy())
        num_samples += X.size(0)
        if num_samples >= num_samples_max:
            break
    feats = np.vstack(feats)
    labels = np.hstack(labels)
    return utils.visualize_feature_space(feats, labels, num_classes=params.num_classes)


if __name__ == '__main__':
    main()
