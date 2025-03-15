"""
Script for computing flops of models.

Usage:
    python calc_flops.py --model resnet18
"""

import argparse

from calflops import calculate_flops
import torch
import models
from losses import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=2,
                        help='input batch size')
    parser.add_argument('--img_channels', type=int, default=3,
                        help='Number of image channels.')
    parser.add_argument('--img_size', type=int, default=32,
                        help='the height / width of the input image to network')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classes in dataset')
    parser.add_argument('--model', required=True,
                        help='baseline | resnet18 | resnet34 | resnet50')
    parser.add_argument('--loss',
                        choices=['ce', 'sphereface', 'cosface', 'arcface', 'curricularface', 'adaface'],
                        default='ce',
                        help='loss function to be used')

    params = parser.parse_args()
    params.use_margin_loss = params.loss not in ['ce', 'focal']
    return params


def main():
    params = parse_args()

    # model
    # try:
    model = getattr(models, params.model)(
        num_classes=params.num_classes, img_channels=params.img_channels)
    # except AttributeError:
    #     raise AttributeError(f'Model {params.model} is not defined')

    # loss
    if params.loss == 'ce':
        pass
    else:
        clf_layer_name = 'fc2' if params.model == 'baseline' else 'fc'
        clf_layer: nn.Linear = model.get_submodule(clf_layer_name)
        if params.loss == 'sphereface':
            criterion = SphereFace(clf_layer)
        elif params.loss == 'cosface':
            criterion = CosFace(clf_layer)
        elif params.loss == 'arcface':
            criterion = ArcFace(clf_layer)
        elif params.loss == 'curricularface':
            criterion = CurricularFace(clf_layer)
        elif params.loss == 'adaface':
            criterion = AdaFace(clf_layer)
        else:
            criterion = None
        del clf_layer
        criterion.reset_parameters()
        model = criterion.patch_model(model, clf_layer_name)
        del criterion

    # run
    input_shape = (params.batch_size, params.img_channels, params.img_size, params.img_size)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=False,
                                          output_precision=4)
    print(f'[Analyzing {model.__class__.__qualname__} model]')
    print(f'FLOPs={flops:.05f}, MACs={macs:.05f}, #Params={params:.05f}')


if __name__ == '__main__':
    main()
