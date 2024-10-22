"""
Training DCT-CryptoNets

author: Arjun Roy <roy208@purdue.edu>
"""
from __future__ import print_function

import os, sys
import torch.nn as nn
import time
import gc

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torchinfo import summary

# Local modules
from io_utils import model_dict, parse_args
from data.datamgr import SimpleDataManager
from utils import *


use_gpu = torch.cuda.is_available()
print(f'Using GPU: {use_gpu}\n')


def train(params, model, optimizer, criterion, train_loader, val_loader, start_epoch, stop_epoch, early_stopping):

    # Grab highest val accuracy if resuming training
    if model.module.best_prec1_val is None:
        best_val_acc = 0
    else:
        best_val_acc = model.module.best_prec1_val

    # Train epochs
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        t = time.time()

        train_loss = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        val_loss = AverageMeter()
        top1_val = AverageMeter()
        top5_val = AverageMeter()

        params = adjust_learning_rate(params, optimizer, epoch)
        print(f'\nEpoch: [{epoch + 1} | {stop_epoch}] LR: {get_lr(optimizer)}')

        for batch_idx, (data, target) in enumerate(train_loader):
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            f, output = model.forward(data)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            train_loss.update(loss.data.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # Compute gradient and perform optimizer step
            optimizer.zero_grad()
            loss.backward()
            if params.grad_clip_value is not None:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=params.grad_clip_value)
            elif params.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.grad_clip_norm, norm_type=2)
            optimizer.step()

            # Display progress
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

        # Validate every epoch
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if use_gpu:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                f, output = model.forward(data)
                loss = criterion(output, target)

                # Measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
                val_loss.update(loss.data.item(), data.size(0))
                top1_val.update(prec1.item(), data.size(0))
                top5_val.update(prec5.item(), data.size(0))

            print(f'Avg. Val Loss: {val_loss.avg:.3f} | Top-1 Acc: {top1_val.avg:.3f}% | Top-5 Acc: {top5_val.avg:.3f}%')

        torch.cuda.empty_cache()
        validation_time = time.time() - t
        print(f'Time for validation epoch {epoch}: {validation_time/60:.2f} minutes')

        # Save best model
        if top1_val.avg > best_val_acc:
            best_val_acc = top1_val.avg
            torch.save({
                'epoch': epoch,
                'state': model.state_dict(),
                'prec1': top1_val.avg,
                'prec5': top5_val.avg,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(params.checkpoint_dir, 'best.tar'))

        # Early stopping
        if early_stopping(val_loss.avg):
            print(f'Early stopping at epoch: {epoch}')
            break

        # Reset validation records each epoch
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
                (f'{params.save_path}/checkpoints/{params.dataset}/{params.model}_dct/'
                 f'filter_{params.filter_size}'
                 f'__pattern_{params.dct_pattern}'
                 f'__input_{params.channels}_{params.image_size_dct}_{params.image_size_dct}'
                 f'__bitwidth_{params.bit_width}')
        else:
            params.checkpoint_dir = \
                (f'{params.save_path}/checkpoints/{params.dataset}/{params.model}/'
                 f'input_{params.channels}_{params.image_size}_{params.image_size}'
                 f'__bitwidth_{params.bit_width}')
    print(f'Checkpoint dir: {params.checkpoint_dir}')
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # Data manager and transformations
    normalize_param = None  # Use default normalization param for ImageNet, miniImageNet and Imagenette in datamgr
    jitter_param = None     # Use default jitter param for ImageNet, miniImageNet and Imagenette in datamgr
    if not params.dct_status and params.dataset == 'cifar10':
        normalize_param = dict(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
        jitter_param = dict(
            Brightness=0.1,
            Contrast=0.1,
            Color=0.1,
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
            aug=params.train_aug,
            filter_size=params.filter_size,
            channels=params.channels,
            dct_pattern=params.dct_pattern,
        )
        test_transform = test_datamgr.trans_loader.get_composed_transform_dct_img(
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

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, sampler=train_sampler, **data_loader_params)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=params.test_batch_size, sampler=val_sampler, **data_loader_params)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.test_batch_size, shuffle=False, **data_loader_params)

    elif params.dataset == 'Imagenette':
        trainset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'train'), transform=train_transform)
        valset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)
        testset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, shuffle=True, **data_loader_params)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=params.test_batch_size, shuffle=False, **data_loader_params)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.test_batch_size, shuffle=False, **data_loader_params)

    elif params.dataset == 'miniImagenet':
        base_file = params.dataset_path + 'base.json'
        test_file = params.dataset_path + 'val.json'
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
            'img_size': params.image_size_dct if params.dct_status else params.image_size,
        }
    else:
        model_arg_dict = {
            'in_channels': params.channels,
            'img_size': params.image_size_dct if params.dct_status else params.image_size,
        }
    model = nn.DataParallel(
        BaselineTrain(
            model_dict[params.model](**model_arg_dict),
            params.num_classes,
        )
    )
    print(f'Number Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print('\n============Model Summary============')
    # Ignore parameter count calculated from torchinfo.summary as it doesn't play well with Brevitas QAT
    # Used solely for understanding network topology and tensor dimension changes
    if params.dct_status:
        summary(
            model.module.feature.to('cpu'),
            input_size=(1, params.channels, params.image_size_dct, params.image_size_dct)
        )
    else:
        summary(
            model.module.feature.to('cpu'),
            input_size=(1, params.channels, params.image_size, params.image_size)
        )

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer and early stopping
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
        )
    early_stopping = EarlyStopper(
        patience=10,
        threshold=0.03
    )

    if params.resume:
        # Load checkpoint.
        print('\nResuming from checkpoint...')
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
    train(params, model, optimizer, criterion, train_loader, val_loader, start_epoch, stop_epoch, early_stopping)

    # Test in the clear
    print('\nTesting on best model...')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = nn.DataParallel(
        BaselineTrain(
            model_dict[params.model](**model_arg_dict),
            params.num_classes,
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
    return


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)
