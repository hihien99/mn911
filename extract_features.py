import argparse
import os

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models
import utils
from losses import *


def parse_args():
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(__file__)

    parser.add_argument('exp', type=str, nargs='?', default=None,
                        help='experiment name')
    parser.add_argument('--dataset', default='cifar10',
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
    parser.add_argument('--class_names', type=utils.tuple_inst(str),
                        default=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    parser.add_argument('--model', required=True,
                        help='baseline | resnet18 | resnet34 | resnet50')
    parser.add_argument('--resume_from', default='',
                        help='path to pre-trained weights (to continue training)')
    parser.add_argument('--loss',
                        choices=['ce', 'focal', 'sphereface', 'cosface', 'arcface', 'curricularface', 'adaface'],
                        default='ce',
                        help='loss function to be used')

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training')
    parser.add_argument('--output_dir', default='outputs',
                        help='folder to output images and model checkpoints')

    parser.add_argument('--vis_dim', type=int, choices=[2, 3], default=3,
                        help='visualization dimensionality')

    params = parser.parse_args()
    params.use_margin_loss = params.loss not in ['ce', 'focal']
    print(params)
    return params


def main():
    params = parse_args()
    params.device = torch.device(params.device)

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
    except AttributeError:
        raise AttributeError(f'Model {params.model} is not defined')

    # loss
    if params.loss != 'ce':
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
        del criterion

    if len(params.resume_from):
        model.load_state_dict(
            torch.load(params.resume_from, weights_only=False, map_location=params.device), strict=False)
        print(f'Loaded weights from {params.resume_from}')
    elif params.eval_only:
        raise ValueError('No weights provided for evaluation')
    print(model)

    # visualize
    visualize_loop(model, train_loader, params)


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
    np.save(os.path.join(params.output_dir, 'feats.npy'), feats)
    np.save(os.path.join(params.output_dir, 'labels.npy'), labels)


if __name__ == '__main__':
    main()
