"""
Code for training both MNIST and Chest X-Ray datasets

author: Arjun Roy <roy208@purdue.edu>
"""
from __future__ import print_function

import os, sys
import torch.nn as nn
import time
import gc
import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

from torchinfo import summary
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

from io_utils import model_dict, parse_args
import configs
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain
from utils import AverageMeter, accuracy


use_gpu = torch.cuda.is_available()
print(f'Using GPU: {use_gpu}\n')


class EarlyStopper:
    """ Early stopping based on validation loss tracking """
    def __init__(self, patience=1, threshold=0.0):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.threshold):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def adjust_learning_rate(params, optimizer, epoch):
    epoch += 1
    if epoch in params.schedule:
        params.lr *= params.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = params.lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def plot_examples(params, train_data):
    rand_idx = random.sample(range(len(train_data)), k=16)
    plt.figure(figsize=(10, 10))

    for i, idx in enumerate(rand_idx):
        img, label = train_data[idx]
        # the image tensor's range is not between 0 and 1,so we have to temporarily scale the tensor values into range 0 and 1 to prevent error.
        img = (img - img.min()) / (img.max() - img.min())
        img_class = train_data.classes[label]

        plt.subplot(4, 4, i + 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Class : {img_class}", fontsize=10)
    plt.savefig(f'{params.checkpoint_dir}/example_images.png', dpi=400)
    return


def pred_classes(params, model, test_data):
    predicted_labels = []
    actual_labels = []

    model.eval()
    with torch.no_grad():  # We are using no_grad instead of inference_mode for better compatibility
        for images, labels in torch.utils.data.DataLoader(dataset=test_data, batch_size=params.test_batch_size):
            images, labels = images.cuda(), labels.cuda()
            f, prediction_logits = model.forward(images)
            predictions = prediction_logits.argmax(dim=1).cpu().numpy()
            predicted_labels.extend(predictions)
            true_labels = labels.cpu().numpy()
            actual_labels.extend(true_labels)

    confusion_mat = confusion_matrix(actual_labels, predicted_labels)
    confusion_df = pd.DataFrame(confusion_mat / np.sum(confusion_mat) * 10,
                                index=test_data.classes,
                                columns=test_data.classes)
    plt.figure(figsize=(12, 7))
    sn.heatmap(confusion_df, annot=True)
    plt.savefig(f'{params.checkpoint_dir}/heatmap.png', dpi=400)
    return


def train(params, model, optimizer, criterion, train_loader, val_loader, start_epoch, stop_epoch, early_stopping, scheduler):

    # grab highest val accuracy if resuming training
    if model.module.best_prec1_val is None:
        best_val_acc = 0
    else:
        best_val_acc = model.module.best_prec1_val

    # train epochs
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        t = time.time()

        train_loss = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        val_loss = AverageMeter()
        top1_val = AverageMeter()
        top5_val = AverageMeter()

        adjust_learning_rate(params, optimizer, epoch)
        print(f'\nEpoch: [{epoch + 1} | {stop_epoch}] LR: {get_lr(optimizer)}')

        for batch_idx, (data, target) in enumerate(train_loader):
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # output = model(data)
            f, output = model.forward(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            train_loss.update(loss.data.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # compute gradient and perform optimizer step
            optimizer.zero_grad()
            loss.backward()
            if params.grad_clip_value is not None:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=params.grad_clip_value)
            elif params.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.grad_clip_norm, norm_type=2)
            optimizer.step()

            # display progress
            if batch_idx % (len(train_loader)//10) == 0:
                print(f'[{batch_idx}/{len(train_loader)}] Avg. Train Loss: {train_loss.avg:.3f} | '
                      f'Top-1 Acc: {top1.avg:.3f}% | Top-5 Acc: {top5.avg:.3f}%')

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            torch.save({
                'epoch': epoch,
                'state': model.state_dict(),
                'prec1': top1.avg,
                'prec5': top5.avg,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch)))

        training_time = time.time() - t
        print(f'Time for training epoch {epoch}: {training_time/60:.2f} minutes')
        t = time.time()

        # validate every epoch
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if use_gpu:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                # output = model(data)
                f, output = model.forward(data)
                loss = criterion(output, target)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
                val_loss.update(loss.data.item(), data.size(0))
                top1_val.update(prec1.item(), data.size(0))
                top5_val.update(prec5.item(), data.size(0))

            print(f'Avg. Val Loss: {val_loss.avg:.3f} | Top-1 Acc: {top1_val.avg:.3f}% | Top-5 Acc: {top5_val.avg:.3f}%')

        torch.cuda.empty_cache()
        validation_time = time.time() - t
        print(f'Time for validation epoch {epoch}: {validation_time/60:.2f} minutes')

        # # update LR with scheduler
        # scheduler.step(val_loss.avg)

        # save best model
        if top1_val.avg > best_val_acc:
            best_val_acc = top1_val.avg
            torch.save({
                'epoch': epoch,
                'state': model.state_dict(),
                'prec1': top1_val.avg,
                'prec5': top5_val.avg,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(params.checkpoint_dir, 'best.tar'))

        # early stopping
        if early_stopping(val_loss.avg):
            print(f'Early stopping at epoch: {epoch}')
            break

        # reset validation records each epoch
        val_loss.reset()
        top1_val.reset()
        top5_val.reset()

    return model


def test(model, criterion, val_loader, test_loader):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # output = model(data)
            f, output = model.forward(data)
            loss = criterion(output, target)
            val_loss += loss.data.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()

        avg_val_loss = val_loss / (batch_idx + 1)
        val_acc = 100. * (correct / total)
        print(f'Avg. Val Loss: {avg_val_loss:.3f} | Acc: {correct}/{total} ({val_acc:.2f}%)')

        test_loss = 0
        correct = 0
        total = 0
        for data, target in test_loader:
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # output = model(data)
            f, output = model.forward(data)
            loss = criterion(output, target)
            test_loss += loss.data.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()

        avg_test_loss = test_loss / (batch_idx + 1)
        test_acc = 100. * (correct / total)
        print(f'Avg. Test Loss: {avg_test_loss:.3f} | Acc: {correct}/{total} ({test_acc:.2f}%)')

    return val_acc, test_acc


def main():
    params = parse_args('train')

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if not params.checkpoint_dir.strip():
        if params.dct_status:
            params.checkpoint_dir = \
                (f'{configs.save_dir}/checkpoints/{params.dataset}/{params.model}_dct/'
                 f'filter_{params.filter_size}_pattern_{params.dct_pattern}'
                 f'_input_{params.channels}_{params.image_size_dct}_{params.image_size_dct}'
                 f'_bitwidth_{params.bit_width}')
        else:
            params.checkpoint_dir = \
                (f'{configs.save_dir}/checkpoints/{params.dataset}/{params.model}/'
                 f'input_{params.channels}_{params.image_size}_{params.image_size}'
                 f'_bitwidth_{params.bit_width}')
    print(f'Checkpoint dir: {params.checkpoint_dir}')
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # Data Manager and transforms
    normalize_param = None
    jitter_param = None
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
            jitter_param = dict(
                Brightness=0.1,
                Contrast=0.1,
                Color=0.1,
            )
        elif params.dataset == 'cifar100':
            normalize_param = dict(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            )

    if params.dct_status:
        train_datamgr = SimpleDataManager(
            params.image_size_dct,
            batch_size=params.batch_size,
            normalize_param=normalize_param,
            jitter_param=jitter_param,
        )
        test_datamgr = SimpleDataManager(
            params.image_size_dct,
            batch_size=params.test_batch_size,
            normalize_param=normalize_param,
            jitter_param=jitter_param,
        )
        train_transform = train_datamgr.trans_loader.get_composed_transform_dct_img(
        # train_transform=train_datamgr.trans_loader.get_composed_transform_dct_np(
            aug=params.train_aug,
            filter_size=params.filter_size,
            channels=params.channels,
            dct_pattern=params.dct_pattern,
        )
        test_transform = test_datamgr.trans_loader.get_composed_transform_dct_img(
        # test_transform=test_datamgr.trans_loader.get_composed_transform_dct_np(
            aug=False,
            filter_size=params.filter_size,
            channels=params.channels,
            dct_pattern=params.dct_pattern,
        )
    else:
        train_datamgr = SimpleDataManager(
            params.image_size,
            batch_size=params.batch_size,
            normalize_param=normalize_param,
            jitter_param=jitter_param,
        )
        test_datamgr = SimpleDataManager(
            params.image_size,
            batch_size=params.test_batch_size,
            normalize_param=normalize_param,
            jitter_param=jitter_param,
        )
        train_transform = train_datamgr.trans_loader.get_composed_transform(aug=params.train_aug)
        test_transform = test_datamgr.trans_loader.get_composed_transform(aug=False)

    # Dataset
    data_loader_params = dict(
        num_workers=params.num_workers,
        pin_memory=True,
    )
    if params.dataset == 'cifar10':
        trainset = datasets.CIFAR10(root=params.dataset_path, train=True, download=True, transform=train_transform)
        valset = datasets.CIFAR10(root=params.dataset_path, train=True, download=True, transform=test_transform)
        testset = datasets.CIFAR10(root=params.dataset_path, train=False, download=True, transform=test_transform)

        num_train = len(trainset)
        train_idx, val_idx = train_test_split(np.arange(num_train), test_size=0.1, random_state=42)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=params.batch_size,
            sampler=train_sampler,
            **data_loader_params,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=params.test_batch_size,
            sampler=val_sampler,
            **data_loader_params,
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=params.test_batch_size,
            shuffle=False,
            **data_loader_params,
        )

    elif params.dataset == 'cifar100':
        trainset = datasets.CIFAR100(root=params.dataset_path, train=True, download=True, transform=train_transform)
        valset = datasets.CIFAR100(root=params.dataset_path, train=True, download=True, transform=test_transform)
        testset = datasets.CIFAR100(root=params.dataset_path, train=False, download=True, transform=test_transform)

        num_train = len(trainset)
        train_idx, val_idx = train_test_split(np.arange(num_train), test_size=0.2, random_state=42)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=params.batch_size,
            sampler=train_sampler,
            **data_loader_params,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=params.test_batch_size,
            sampler=val_sampler,
            **data_loader_params,
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=params.test_batch_size,
            shuffle=False,
            **data_loader_params,
        )

    elif params.dataset == 'Imagenette':
        trainset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'train'), transform=train_transform)
        valset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)
        testset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, shuffle=True, **data_loader_params)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=params.test_batch_size, shuffle=False, **data_loader_params)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.test_batch_size, shuffle=False, **data_loader_params)

    elif params.dataset == 'mnist':
        trainset = datasets.MNIST(root=params.dataset_path, train=True, download=True, transform=train_transform)
        valset = datasets.MNIST(root=params.dataset_path, train=True, download=True, transform=test_transform)
        testset = datasets.MNIST(root=params.dataset_path, train=False, download=True, transform=test_transform)

        num_train = len(trainset)
        train_idx, val_idx = train_test_split(np.arange(num_train), test_size=0.2, random_state=42)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=params.batch_size,
            sampler=train_sampler,
            **data_loader_params,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=params.test_batch_size,
            sampler=val_sampler,
            **data_loader_params,
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=params.test_batch_size,
            shuffle=False,
            **data_loader_params,
        )

    elif params.dataset == 'chest_xray':
        trainset = datasets.ImageFolder(os.path.join(params.dataset_path, 'train'), transform=train_transform)
        valset = datasets.ImageFolder(os.path.join(params.dataset_path, 'val'), transform=test_transform)
        testset = datasets.ImageFolder(os.path.join(params.dataset_path, 'test'), transform=test_transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, shuffle=True, **data_loader_params)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=params.test_batch_size, shuffle=False, **data_loader_params)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.test_batch_size, shuffle=False, **data_loader_params)

    elif params.dataset == 'miniImagenet':
        base_file = configs.data_dir[params.dataset] + 'base.json'
        test_file = configs.data_dir[params.dataset] + 'val.json'
        if params.dct_status:
            base_datamgr = SimpleDataManager(params.image_size_dct, batch_size=params.batch_size)
            train_loader, trainset = base_datamgr.get_data_loader_dct(base_file, aug=params.train_aug, filter_size=params.filter_size, channels=params.channels)
            base_datamgr_val = SimpleDataManager(params.image_size_dct, batch_size=params.test_batch_size)
            val_loader, valset = base_datamgr_val.get_data_loader_dct(base_file, aug=False, filter_size=params.filter_size, channels=params.channels)
            test_datamgr = SimpleDataManager(params.image_size_dct, batch_size=params.test_batch_size)
            test_loader, testset = test_datamgr.get_data_loader_dct(test_file, aug=False, filter_size=params.filter_size, channels=params.channels)
        else:
            base_datamgr = SimpleDataManager(params.image_size, batch_size=params.batch_size)
            train_loader, trainset = base_datamgr.get_data_loader(base_file , aug=params.train_aug)
            base_datamgr_val = SimpleDataManager(params.image_size, batch_size=params.test_batch_size)
            val_loader, valset = base_datamgr_val.get_data_loader(base_file, aug=False)
            test_datamgr = SimpleDataManager(params.image_size, batch_size=params.test_batch_size)
            test_loader, testset = test_datamgr.get_data_loader(test_file, aug=False)

    elif params.dataset == 'ImageNet':
        trainset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'train'), transform=train_transform)
        valset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)
        testset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, shuffle=True, **data_loader_params)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=params.test_batch_size, shuffle=False, **data_loader_params)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.test_batch_size, shuffle=False, **data_loader_params)

    # Model
    if 'qat' in params.model:
        model_arg_dict = {
            'bit_width': params.bit_width,
            'in_channels': params.channels,
            # 'pruning_percent': params.pruning,
            'img_size': params.image_size_dct if params.dct_status else params.image_size,
        }
    elif params.model == 'MobileNetv2':
        model_arg_dict = {
            'num_classes': params.num_classes,
            'input_size': params.image_size_dct if params.dct_status else params.image_size,
        }
    elif params.model == 'MobileNetv2DCT':
        model_arg_dict = {
            'channels': params.channels,
            'num_classes': params.num_classes,
            # 'input_size': params.image_size_dct if params.dct_status else params.image_size,
        }
    else:
        model_arg_dict = {
            'in_channels': params.channels,
            # 'pruning_percent': params.pruning,
            'img_size': params.image_size_dct if params.dct_status else params.image_size,
        }
    model = nn.DataParallel(
        # resnet20()
        # torchvision.models.resnet20(pretrained=False, num_classes=10),
        # torchvision.models.resnet18(pretrained=False, num_classes=10),
        # model_dict[params.model](**model_arg_dict),       # For MobileNetV2
        BaselineTrain(                                      # For all backbone models
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

    # Loss
    if params.dataset == 'xray':
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    if params.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay
        )
    elif params.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay
        )
    elif params.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=params.lr,
            momentum=params.momentum,
            weight_decay=params.weight_decay,
            # nesterov=True,
        )

    # lr scheduler and early stopping
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=5,
        cooldown=3,
        threshold=0.01,
        mode='min',
    )
    early_stopping = EarlyStopper(
        patience=10,
        threshold=0.03
    )

    if params.resume:
        # Load checkpoint.
        print('Resuming from checkpoint...')
        checkpoint = torch.load(params.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = params.lr
        model.module.best_prec1_val = checkpoint["prec1"]
        print(f'Loaded checkpoint {params.resume} ({model.module.best_prec1_val:.3f}% Top-1 Acc. @ epoch {checkpoint["epoch"]})')

    if use_gpu:
        model.cuda()
        criterion.cuda()

    # Train
    if not params.dct_status:
        plot_examples(params, trainset)
    model.module.classifier.register_forward_hook(lambda m, inp, out: F.dropout(out, p=params.dropout, training=True))
    train(params, model, optimizer, criterion, train_loader, val_loader, start_epoch, stop_epoch, early_stopping, scheduler)

    # Test
    print('\nTesting on best model...')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = nn.DataParallel(
        # model_dict[params.model](**model_arg_dict),   # For MobileNetV2
        BaselineTrain(                                  # For all backbone models
            model_dict[params.model](**model_arg_dict),
            params.num_classes,
            loss_type='softmax'
        )
    )
    checkpoint = torch.load(f'{params.checkpoint_dir}/best.tar')
    model.load_state_dict(checkpoint['state'])
    print(f'Loaded best model {params.checkpoint_dir}/best.tar (epoch {checkpoint["epoch"]})')
    model.module.classifier.register_forward_hook(lambda m, inp, out: F.dropout(out, p=params.dropout, training=False))
    if use_gpu:
        model.cuda()
        criterion.cuda()
    test(model, criterion, val_loader, test_loader)
    if params.dataset == 'cifar10' or params.dataset == 'Imagenette':
        pred_classes(params, model, testset)
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
