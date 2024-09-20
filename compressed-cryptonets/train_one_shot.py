#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
"""
Code for training 1-shot datasets (mainly for mini-ImageNet)

"""

from __future__ import print_function

import argparse
import csv
import os, sys
import time

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.datamgr import SimpleDataManager, SetDataManager  # ,SimpleDataManager_dct , SetDataManager_dct
import configs
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
# import wrn_mixup_model, res_mixup_model
from io_utils import model_dict, parse_args, get_resume_file, get_assigned_file, get_best_file
from os import path

use_gpu = torch.cuda.is_available()
print(f'Using GPU: {use_gpu}\n')


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


early_stopping = EarlyStopping(tolerance=5, min_delta=0.1)


def train_baseline(base_loader, base_loader_test, val_loader, model, start_epoch, stop_epoch, params, tmp):
    if params.dct_status:
        channels = params.channels
    else:
        channels = 3

    val_acc_best = 0.0

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    if path.exists(params.checkpoint_dir + '/val_' + params.dataset + '.pt'):
        loader = torch.load(params.checkpoint_dir + '/val_' + params.dataset + '.pt')
    else:
        loader = []
        for ii, (x, _) in enumerate(val_loader):
            loader.append(x)
            # print("head of train_dct: ", x.shape)
        torch.save(loader, params.checkpoint_dir + '/val_' + params.dataset + '.pt')

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters())
    print("stop_epoch", start_epoch, stop_epoch)
    t = time.time()
    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        correct1 = 0.0
        total = 0

        for batch_idx, (input_var, target_var) in enumerate(base_loader):
            # print(f'Image pixel range [{torch.min(input_var)}:{torch.max(input_var)}]')
            if use_gpu:
                input_var, target_var = input_var.cuda(), target_var.cuda()
            input_dct_var, target_var = Variable(input_var), Variable(target_var)
            # print(f'Image pixel range [{torch.min(input_dct_var)}:{torch.max(input_dct_var)}]')
            # f, outputs = model.forward(input_dct_var, training=True)
            f, outputs = model.forward(input_dct_var)
            loss = criterion(outputs, target_var)
            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target_var.size(0)
            correct += predicted.eq(target_var.data).cpu().sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('{0}/{1}'.format(batch_idx, len(base_loader)), 'Loss: %.3f | Acc: %.3f%%  '
                      % (train_loss / (batch_idx + 1), 100. * correct / total))

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        elapsed_time = time.time() - t
        print(f"Time for training epoch {epoch}: {elapsed_time/60:.2f} minutes")

        # Test and Validation every epoch
        model.eval()
        with torch.no_grad():
            # Test Set
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(base_loader_test):
                if use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                # f, outputs = model.forward(inputs, training=False)
                f, outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.data.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            print('Test Loss: %.3f | Acc: %.3f%%'
                  % (test_loss / (batch_idx + 1), 100. * correct / total))

            # # Validation Set
            # # validation set is out of distribution from train/test
            # val_loss = 0
            # correct = 0
            # total = 0
            # for batch_idx, (inputs, targets) in enumerate(val_loader):
            #     if use_gpu:
            #         inputs, targets = inputs.cuda(), targets.cuda()
            #     inputs, targets = Variable(inputs), Variable(targets)
            #     f, outputs = model.forward(inputs)
            #     loss = criterion(outputs, targets)
            #     val_loss += loss.data.item()
            #     _, predicted = torch.max(outputs.data, 1)
            #     total += targets.size(0)
            #     correct += predicted.eq(targets.data).cpu().sum()
            #
            # print('Val Loss: %.3f | Acc: %.3f%%'
            #       % (val_loss / (batch_idx + 1), 100. * correct / total))
            torch.cuda.empty_cache()

            # Early Stopping
            early_stopping(train_loss, test_loss)
            if early_stopping.early_stop:
                print(f"We stop at epoch: {epoch}")
                break

            # valmodel = BaselineFinetune(model_dict[params.model], params.train_n_way, params.n_shot, loss_type='softmax')
            # valmodel.n_query = 15
            # acc_all1, acc_all2, acc_all3 = [], [], []
            # for i, x in enumerate(loader):
            #     val_loss = 0
            #     if params.dct_status:
            #         x = x.view(-1, channels, image_size_dct, image_size_dct)
            #     else:
            #         x = x.view(-1, channels, image_size, image_size)
            #
            #     if use_gpu:
            #         x = x.cuda()
            #         i = i.cuda()
            #
            #     with torch.no_grad():
            #         f, scores = model(x)
            #     f = f.view(params.train_n_way, params.n_shot+valmodel.n_query, -1)
            #     scores = valmodel.set_forward_adaptation(f.cpu())
            #     loss = criterion(scores, i)
            #     val_loss += loss.data.item()
            #     acc = []
            #     for each_score in scores:
            #         pred = each_score.data.cpu().numpy().argmax(axis=1)
            #         y = np.repeat(range(5), 15)
            #         acc.append(np.mean(pred == y)*100)
            #     acc_all1.append(acc[0])
            #     acc_all2.append(acc[1])
            #     acc_all3.append(acc[2])
            #
            # print('Test Acc at 100= %4.2f%%' % (np.mean(acc_all1)))
            # print('Test Acc at 200= %4.2f%%' % (np.mean(acc_all2)))
            # print('Test Acc at 300= %4.2f%%' % (np.mean(acc_all3)))
            #
            # if np.mean(acc_all3) > val_acc_best:
            #     val_acc_best = np.mean(acc_all3)
            #     bestfile = os.path.join(params.checkpoint_dir, 'best.tar')
            #     torch.save({'epoch': epoch, 'state': model.state_dict()}, bestfile)

    # model.module.module.feature.unprune()
    final_file = os.path.join(params.checkpoint_dir, 'final.tar')
    torch.save({'epoch': epoch, 'state': model.state_dict()}, final_file)
    return model


if __name__ == '__main__':
    params = parse_args('train')

    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'
    if params.dct_status == False:
        params.channels = 3
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s_%sway_%sshot' % (
    configs.save_dir, params.dataset, params.model, params.method, params.train_n_way, params.n_shot)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    if params.dct_status:
        params.checkpoint_dir += '_dct'

    if params.filter_size != 8:
        params.checkpoint_dir += '_%sfiltersize' % (params.filter_size)

    image_size = params.image_size
    image_size_dct = params.image_size_dct
    params.num_classes = 200
    if params.dataset == 'cifar':
        image_size = 32
        params.num_classes = 64
    else:
        if params.model == 'WideResNet28_10':
            image_size = 84
            params.num_classes = 200

    print(params.checkpoint_dir)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method in ['baseline++', 'S2M2_R', 'rotation']:
        if params.dct_status:
            base_datamgr = SimpleDataManager(
                image_size_dct,
                batch_size=params.batch_size,
            )
            base_datamgr_test = SimpleDataManager(
                image_size_dct,
                batch_size=params.test_batch_size
            )
            val_datamgr = SimpleDataManager(
                image_size_dct,
                batch_size=params.test_batch_size
            )
            base_loader = base_datamgr.get_data_loader_dct(
                base_file,
                aug=params.train_aug,
                filter_size=params.filter_size,
                channels=params.channels,
            )
            base_loader_test = base_datamgr_test.get_data_loader_dct(
                base_file,
                aug=False,
                filter_size=params.filter_size,
                channels=params.channels,
            )
            val_loader = val_datamgr.get_data_loader_dct(
                val_file,
                aug=False,
                filter_size=params.filter_size,
                channels=params.channels,
            )
        else:
            base_datamgr = SimpleDataManager(image_size, batch_size=params.batch_size)
            base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
            base_datamgr_test = SimpleDataManager(image_size, batch_size=params.test_batch_size)
            base_loader_test = base_datamgr_test.get_data_loader(base_file, aug=False)
            val_datamgr = SimpleDataManager(image_size, batch_size=params.test_batch_size)
            val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        if params.method == 'baseline++':
            # model = nn.DataParallel(BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist'))
            model = nn.DataParallel(BaselineTrain(model_dict[params.model](in_channels=params.channels), params.num_classes, loss_type='softmax'))
            print(f'Number Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
            if params.dct_status:
                summary(model.module.feature.to('cpu'),
                        input_size=(1, params.channels, params.image_size_dct, params.image_size_dct))
            else:
                summary(model.module.feature.to('cpu'),
                        input_size=(1, params.channels, params.image_size, params.image_size))

    if params.method == 'baseline++':
            if use_gpu:
                # if torch.cuda.device_count() > 1:
                #     model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
                model.cuda()

            if params.resume:
                resume_file = get_resume_file(params.checkpoint_dir)
                tmp = torch.load(resume_file)
                start_epoch = tmp['epoch'] + 1
                state = tmp['state']
                model.load_state_dict(state)
            model = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True
            optimization = 'Adam'
            model = train_baseline(base_loader, base_loader_test, val_loader, model, start_epoch,
                                   start_epoch + stop_epoch, params, {})
