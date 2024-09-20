"""
Code for training both MNIST and Chest X-Ray datasets

author: Arjun Roy <roy208@purdue.edu>
"""
from __future__ import print_function

import argparse
import os, sys
import torch.nn as nn
import time

import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchinfo import summary
import numpy as np

from io_utils import model_dict, parse_args
import configs
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain


use_gpu = torch.cuda.is_available()
print(f'Using GPU: {use_gpu}\n')


def test(model, criterion, test_loader):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        for data, target in test_loader:
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # output = model(data)
            f, output = model.forward(data)
            # test_loss += criterion(output, target.unsqueeze(1).float()).item() # sum up batch loss
            loss = criterion(output, target)
            test_loss += loss.data.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()

        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, total,
            100. * correct / total)
        )


def main():
    params = parse_args('test')

    # Data Manager and transforms
    normalize_param = None
    if not params.dct_status:
        if params.dataset == 'mnist':
            normalize_param = dict(
                mean=[0.1307,],
                std=[0.3081,]
            )
        elif params.dataset == 'cifar10':
            normalize_param = dict(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            )
        elif params.dataset == 'cifar100':
            normalize_param = dict(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            )

    if params.dct_status:
        test_datamgr = SimpleDataManager(
            params.image_size_dct,
            batch_size=params.test_batch_size,
            normalize_param=normalize_param,
        )
        test_transform = test_datamgr.trans_loader.get_composed_transform_dct_img(
            aug=False,
            filter_size=params.filter_size,
            channels=params.channels,
            dct_pattern=params.dct_pattern,
        )
    else:
        test_datamgr = SimpleDataManager(
            params.image_size,
            batch_size=params.test_batch_size,
            normalize_param=normalize_param,
        )
        test_transform = test_datamgr.trans_loader.get_composed_transform(aug=False)

    # Dataset
    if params.dataset == 'cifar10':
        trainset = datasets.CIFAR10(root=params.dataset_path, train=True, download=True, transform=test_transform)
        valset = datasets.CIFAR10(root=params.dataset_path, train=True, download=True, transform=test_transform)
        testset = datasets.CIFAR10(root=params.dataset_path, train=False, download=True, transform=test_transform)

        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))
        train_idx, val_idx = indices[split:], indices[:split]
        val_sampler = SubsetRandomSampler(val_idx)

        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=params.test_batch_size,
            sampler=val_sampler,
        )
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.test_batch_size, shuffle=False)

    elif params.dataset == 'cifar100':
        trainset = datasets.CIFAR100(root=params.dataset_path, train=True, download=True, transform=test_transform)
        valset = datasets.CIFAR100(root=params.dataset_path, train=True, download=True, transform=test_transform)
        testset = datasets.CIFAR100(root=params.dataset_path, train=False, download=True, transform=test_transform)

        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))
        train_idx, val_idx = indices[split:], indices[:split]
        val_sampler = SubsetRandomSampler(val_idx)

        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=params.test_batch_size,
            sampler=val_sampler,
        )
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.test_batch_size, shuffle=False)

    elif params.dataset == 'Imagenette':
        valset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)
        testset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)

        val_loader = torch.utils.data.DataLoader(valset, batch_size=params.test_batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.test_batch_size, shuffle=False)

    elif params.dataset == 'mnist':
        trainset = datasets.MNIST(root=params.dataset_path, train=True, download=True, transform=test_transform)
        valset = datasets.MNIST(root=params.dataset_path, train=True, download=True, transform=test_transform)
        testset = datasets.MNIST(root=params.dataset_path, train=False, download=True, transform=test_transform)

        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))
        train_idx, val_idx = indices[split:], indices[:split]
        val_sampler = SubsetRandomSampler(val_idx)

        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=params.test_batch_size,
            sampler=val_sampler,
        )
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.test_batch_size, shuffle=False)

    elif params.dataset == 'chest_xray':
        valset = datasets.ImageFolder(os.path.join(params.dataset_path, 'val'), transform=test_transform)
        testset = datasets.ImageFolder(os.path.join(params.dataset_path, 'test'), transform=test_transform)

        val_loader = torch.utils.data.DataLoader(valset, batch_size=params.test_batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.test_batch_size, shuffle=False)

    # Model, loss, and optimizer
    if 'qat' in params.model:
        model_arg_dict = {
            'bit_width': params.bit_width,
            'in_channels': params.channels,
            'pruning_percent': params.pruning,
            'img_size': params.image_size_dct if params.dct_status else params.image_size,
        }
    # elif params.model == 'MobileNetv2':
    #     model_arg_dict = {
    #         'n_class': params.num_classes,
    #         'input_size': params.image_size_dct if params.dct_status else params.image_size,
    #     }
    # elif params.model == 'MobileNetv2DCT':
    #     model_arg_dict = {
    #         'channels': params.channels,
    #     }
    else:
        model_arg_dict = {
            'in_channels': params.channels,
            'pruning_percent': params.pruning,
            'img_size': params.image_size_dct if params.dct_status else params.image_size,
        }
    model = nn.DataParallel(
        # torchvision.models.resnet18(pretrained=False, num_classes=10),
        # model_dict[params.model](**model_arg_dict)
        BaselineTrain(
            model_dict[params.model](**model_arg_dict),
            params.num_classes,
            loss_type='softmax'
        )
    )
    print(f'Number Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print('\n============Model Summary============')
    if params.dct_status:
        summary(
            model.module.feature.to('cpu'),
            # model.to('cpu'),
            input_size=(1, params.channels, params.image_size_dct, params.image_size_dct)
        )
    else:
        summary(
            model.module.feature.to('cpu'),
            # model.to('cpu'),
            input_size=(1, params.channels, params.image_size, params.image_size)
        )

    # Loss and classifier
    if params.dataset == 'xray':
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Load checkpoint.
    print('Resuming from checkpoint...')
    checkpoint = torch.load(params.checkpoint_path)
    model.load_state_dict(checkpoint['state'])
    model.module.feature.unprune()
    print(f'Loaded checkpoint {params.checkpoint_path} (epoch {checkpoint["epoch"]})')

    if use_gpu:
        model.cuda()
        criterion.cuda()
    print('\nValidation Set:')
    test(model, criterion, val_loader)
    print('\nTest Set:')
    test(model, criterion, test_loader)
    print('Done')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)
