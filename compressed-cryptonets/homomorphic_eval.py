from __future__ import print_function

# import argparse
# import csv
import os, sys
from tqdm import tqdm
import time
import pickle

import numpy as np
import torch
from torch.autograd import Variable
# import torch.backends.cudnn as cudnn
import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.datamgr import SimpleDataManager, SetDataManager  #,SimpleDataManager_dct , SetDataManager_dct
import configs
from methods.baselinetrain import BaselineTrain
# from methods.baselinefinetune import BaselineFinetune
# import wrn_mixup_model, res_mixup_model
from io_utils import model_dict, parse_args, get_resume_file ,get_assigned_file, get_best_file
# from os import path
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import onnx
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

from concrete.ml.torch.compile import compile_torch_model, compile_brevitas_qat_model, compile_onnx_model, compile_onnx_model
from concrete.fhe import Configuration
from memory_profiler import profile

from models.polykervnet import Model
from models.pkn import convwrapper, PKN2d
from utils import accuracy, AverageMeter


use_gpu = torch.cuda.is_available()
print(f'Using GPU: {use_gpu}\n')


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


@torch.no_grad()
def test(model, criterion, val_loader, test_loader):
    model.eval()
    with torch.no_grad():
        val_loss = AverageMeter()
        top1_val = AverageMeter()
        top5_val = AverageMeter()
        for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            f, output = model.forward(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            val_loss.update(loss.data.item(), data.size(0))
            top1_val.update(prec1.item(), data.size(0))
            top5_val.update(prec5.item(), data.size(0))

        print(f'Avg. Val Loss: {val_loss.avg:.3f} | '
              f'Top-1 Acc: {top1_val.avg:.3f}% | Top-5 Acc: {top5_val.avg:.3f}%')

        # test_loss = AverageMeter()
        # top1_test = AverageMeter()
        # top5_test = AverageMeter()
        # for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
        #     if use_gpu:
        #         data, target = data.cuda(), target.cuda()
        #     data, target = Variable(data), Variable(target)
        #     f, output = model.forward(data)
        #     loss = criterion(output, target)
        #
        #     # measure accuracy and record loss
        #     prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        #     test_loss.update(loss.data.item(), data.size(0))
        #     top1_test.update(prec1.item(), data.size(0))
        #     top5_test.update(prec5.item(), data.size(0))
        #
        # print(f'Avg. Test Loss: {test_loss.avg:.3f} | '
        #       f'Top-1 Acc: {top1_test.avg:.3f}% | Top-5 Acc: {top5_test.avg:.3f}%')

    return top1_val, top5_val


# @profile
@torch.no_grad()
def test_with_concrete(params, model, data_loader, use_sim, cls):
    top1 = AverageMeter()
    top5 = AverageMeter()

    # # Casting the inputs into int64 is recommended
    # all_y_pred = np.zeros((len(data_loader)), dtype=np.int64)
    # all_targets = np.zeros((len(data_loader)), dtype=np.int64)

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    idx = 0
    for data, target in tqdm(data_loader):
        fhe_mode = "simulate" if use_sim else "execute"
        data = data.numpy()

        # Quantize the inputs and cast to appropriate data type
        encoder_output = model.forward(data, fhe=fhe_mode)

        # Run through clear-text classifier
        try:
            output = cls.forward(torch.from_numpy(encoder_output).flatten().float())  # softmax
        except:
            output = cls.forward(torch.from_numpy(encoder_output).float())  # distLinear

        # measure accuracy and record loss
        if params.test_batch_size == 1:
            prec1, prec5 = accuracy(output.data.view(1, -1), target.data, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        top1.update(prec1.item(), data.shape[0])
        top5.update(prec5.item(), data.shape[0])

    #     endidx = idx + target.shape[0]
    #     # Accumulate the ground truth labels
    #     all_targets[idx:endidx] = target
    #
    #     # Get the predicted class id and accumulate the predictions
    #     y_pred = np.argmax(y_pred.numpy())
    #     all_y_pred[idx:endidx] = y_pred
    #
    #     # Update the index
    #     idx += target.shape[0]
    #
    # # Compute and report results
    # n_correct = np.sum(all_targets == all_y_pred)

    return top1, top5


# @profile
def main():
    # Initializations
    params = parse_args('homomorphic_eval')
    device = torch.device('cpu')

    # Check Quantization type
    if 'QAT' in str(params.model) or 'qat' in str(params.model):
        quantization_type = 'QAT'
    else:
        quantization_type = 'PTQ'

    # Data Manager and transforms
    normalize_param = None
    jitter_param = None
    if not params.dct_status:
        if params.dataset == 'mnist':
            normalize_param = dict(
                mean=[0.1307, ],
                std=[0.3081, ]
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
    if params.dataset == 'cifar10':
        trainset = datasets.CIFAR10(root=params.dataset_path, train=True, download=True, transform=test_transform)
        calibset = datasets.CIFAR10(root=params.dataset_path, train=True, download=True, transform=test_transform)
        valset = datasets.CIFAR10(root=params.dataset_path, train=True, download=True, transform=test_transform)
        testset = datasets.CIFAR10(root=params.dataset_path, train=False, download=True, transform=test_transform)

        num_train = len(trainset)
        train_idx, val_idx = train_test_split(np.arange(num_train), test_size=params.test_subset, random_state=42)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        num_test = len(testset)
        _, test_idx = train_test_split(np.arange(num_test), test_size=params.test_batch_size, random_state=42)
        test_sampler = SubsetRandomSampler(test_idx)

        _, execution_idx = train_test_split(np.arange(num_test), test_size=params.test_subset, random_state=42)
        execution_sampler = SubsetRandomSampler(execution_idx)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=params.batch_size,
            sampler=train_sampler,
        )
        calib_loader = torch.utils.data.DataLoader(
            calibset,
            batch_size=params.batch_size,
            shuffle=False,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=params.test_batch_size,
            sampler=val_sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=params.test_batch_size,
            sampler=test_sampler
        )
        execution_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=params.test_subset,
            sampler=execution_sampler
        )

    elif params.dataset == 'cifar100':
        trainset = datasets.CIFAR100(root=params.dataset_path, train=True, download=True, transform=test_transform)
        valset = datasets.CIFAR100(root=params.dataset_path, train=True, download=True, transform=test_transform)
        testset = datasets.CIFAR100(root=params.dataset_path, train=False, download=True, transform=test_transform)

        num_train = len(trainset)
        train_idx, val_idx = train_test_split(np.arange(num_train), test_size=params.test_subset, random_state=42)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        num_test = len(testset)
        _, test_idx = train_test_split(np.arange(num_test), test_size=params.test_subset, random_state=42)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=params.batch_size,
            sampler=train_sampler,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=params.test_batch_size,
            sampler=val_sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=params.test_batch_size,
            sampler=test_sampler
        )

    elif params.dataset == 'Imagenette':
        trainset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'train'), transform=test_transform)
        calibset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)
        valset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)
        testset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)

        num_train = len(trainset)
        train_idx, val_idx = train_test_split(np.arange(num_train), test_size=params.test_subset, random_state=42)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        num_test = len(testset)
        _, test_idx = train_test_split(np.arange(num_test), test_size=params.test_subset, random_state=42)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=params.batch_size,
            sampler=train_sampler,
        )
        calib_loader = torch.utils.data.DataLoader(
            calibset,
            batch_size=params.batch_size,
            shuffle=False,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=params.test_batch_size,
            sampler=val_sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=params.test_batch_size,
            sampler=test_sampler
        )

    elif params.dataset == 'mnist':
        trainset = datasets.MNIST(root=params.dataset_path, train=True, download=True, transform=test_transform)
        valset = datasets.MNIST(root=params.dataset_path, train=True, download=True, transform=test_transform)
        testset = datasets.MNIST(root=params.dataset_path, train=False, download=True, transform=test_transform)
        testset = torch.utils.data.Subset(testset, list(range(0, params.test_subset, 2)))

        num_train = len(trainset)
        train_idx, val_idx = train_test_split(np.arange(num_train), test_size=params.test_subset, random_state=42)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        num_test = len(testset)
        _, test_idx = train_test_split(np.arange(num_test), test_size=params.test_subset, random_state=42)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=params.batch_size,
            sampler=train_sampler,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=params.test_batch_size,
            sampler=val_sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=params.test_batch_size,
            sampler=test_sampler
        )

    elif params.dataset == 'chest_xray':
        trainset = datasets.ImageFolder(os.path.join(params.dataset_path, 'train'), transform=test_transform)
        valset = datasets.ImageFolder(os.path.join(params.dataset_path, 'val'), transform=test_transform)
        testset = datasets.ImageFolder(os.path.join(params.dataset_path, 'test'), transform=test_transform)

        num_train = len(trainset)
        train_idx, val_idx = train_test_split(np.arange(num_train), test_size=params.test_subset, random_state=42)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        num_test = len(testset)
        _, test_idx = train_test_split(np.arange(num_test), test_size=params.test_subset, random_state=42)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=params.batch_size,
            sampler=train_sampler,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=params.test_batch_size,
            sampler=val_sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=params.test_batch_size,
            sampler=test_sampler
        )

    elif params.dataset == 'miniImagenet':
        base_file = configs.data_dir[params.dataset] + 'base.json'
        test_file = configs.data_dir[params.dataset] + 'val.json'
        if params.dct_status:
            base_datamgr = SimpleDataManager(params.image_size_dct, batch_size=params.batch_size)
            train_loader, trainset = base_datamgr.get_data_loader_dct(base_file, aug=False, filter_size=params.filter_size, channels=params.channels)
            base_datamgr_val = SimpleDataManager(params.image_size_dct, batch_size=params.test_batch_size)
            calib_loader, valset = base_datamgr_val.get_data_loader_dct(base_file, aug=False, filter_size=params.filter_size, subset=params.test_subset, channels=params.channels)
            val_loader, valset = base_datamgr_val.get_data_loader_dct(base_file, aug=False, filter_size=params.filter_size, subset=params.test_subset, channels=params.channels)
            test_datamgr = SimpleDataManager(params.image_size_dct, batch_size=params.test_batch_size)
            test_loader, testset = test_datamgr.get_data_loader_dct(test_file, aug=False, filter_size=params.filter_size, subset=params.test_subset, channels=params.channels)
        else:
            base_datamgr = SimpleDataManager(params.image_size, batch_size=params.batch_size)
            train_loader, trainset = base_datamgr.get_data_loader(base_file, aug=False)
            base_datamgr_val = SimpleDataManager(params.image_size, batch_size=params.test_batch_size)
            calib_loader, valset = base_datamgr_val.get_data_loader(base_file, aug=False, subset=params.test_subset)
            val_loader, valset = base_datamgr_val.get_data_loader(base_file, aug=False, subset=params.test_subset)
            test_datamgr = SimpleDataManager(params.image_size, batch_size=params.test_batch_size)
            test_loader, testset = test_datamgr.get_data_loader(test_file, aug=False, subset=params.test_subset)

    elif params.dataset == 'ImageNet':
        trainset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'train'), transform=test_transform)
        calibset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)
        valset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)
        testset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)

        num_test = len(testset)
        _, test_idx = train_test_split(np.arange(num_test), test_size=params.test_subset, random_state=42)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=params.batch_size,
            shuffle=True,
        )
        calib_loader = torch.utils.data.DataLoader(
            calibset,
            batch_size=params.batch_size,
            shuffle=False,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=params.test_batch_size,
            shuffle=False,
            sampler=test_sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=params.test_batch_size,
            shuffle=False,
            sampler=test_sampler,
        )

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

    # Load checkpoint
    print('Loading checkpoint...')
    checkpoint = torch.load(params.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state'])
    model.module.best_prec1_val = checkpoint["prec1"]
    print(f'Loaded checkpoint {params.checkpoint_path} ({model.module.best_prec1_val:.3f}% Top-1 Acc. @ epoch {checkpoint["epoch"]})')
    # model.module.classifier.register_forward_hook(lambda m, inp, out: F.dropout(out, p=params.dropout, training=False))

    # ## OLD
    # # state = checkpoint['state']
    # state_keys = list(checkpoint.keys())
    # print(state_keys)
    # callwrap = False
    # if 'module' in state_keys[0]:
    #     callwrap = True
    # if callwrap:
    #     model = WrappedModel(model)
    # model_dict_load = model.state_dict()
    # model_dict_load.update(checkpoint)
    # model.load_state_dict(model_dict_load)
    # # model.module.module.feature.unprune()

    # Test model in non-FHE mode
    print(f'\nTesting model in the clear on subset of {params.test_subset} images...')
    model.cuda()
    # test(model, criterion, val_loader, test_loader)

    # Create post-trained quantization calibration data which is first batch of train data
    for data, _ in calib_loader:
        calib_data = data.to(device)
        break
    print(f'Image pixel range [{torch.min(calib_data)}:{torch.max(calib_data)}]')

    # Use only the model features for FHE model
    #  BaselineTrain includes loss function, etc. which cannot be quantized
    # model_feature = model.module.module.feature
    # model_feature.unprune()
    # model_feature = model.model
    # print(model_feature)
    # model_feature = model.module.feature
    # print(model_feature)

    # Create FHE model
    print(f'rounding_threshold_bits: {params.rounding_threshold_bits}')
    print(f'n_bits: {params.n_bits}')
    print(f'p_error: {params.p_error}')
    print(f'Compiling FHE Model (this can take up to half an hour)...')
    model.to(device)
    configuration = Configuration(
        # To enable displaying progressbar
        show_progress=False,
        # To enable showing tags in the progressbar (does not work in notebooks)
        progress_tag=True,
        # To give a title to the progressbar
        progress_title='Evaluation: ',
    )
    t = time.time()
    if quantization_type == 'QAT':
        q_module = compile_brevitas_qat_model(
            model.module.feature,
            calib_data,
            rounding_threshold_bits=params.rounding_threshold_bits,
            n_bits=params.n_bits,
            p_error=params.p_error,
            configuration=configuration,
            verbose=True,
        )
    elif quantization_type == 'PTQ':
        q_module = compile_torch_model(
            model.module.feature,
            calib_data,
            rounding_threshold_bits=6,
            # p_error=0.5,
            n_bits=3,
            configuration=configuration,
            verbose=True,
        )
        # q_module = compile_onnx_model(
        #     onnx.load("/home/arjunroy/Repos/PolyKervNets/polykernet2.onnx"),
        #     calib_data,
        #     rounding_threshold_bits=6,
        #     # p_error=0.5,
        #     n_bits=3,
        #     configuration=configuration,
        #     verbose=True,
        # )
    elapsed_time = time.time() - t
    del calib_data
    print(f"Time for FHE compilation {elapsed_time:.2f}")
    return

    # Check that the network is compatible with FHE constraints
    bitwidth = q_module.fhe_circuit.graph.maximum_integer_bit_width()
    print(
        f"Max bit-width: {bitwidth} bits" + " -> it works in FHE!!"
        if bitwidth <= 16
        else " too high for FHE computation"
    )

    # # Generate MLIR
    # with open('mlir.txt', 'a') as f:
    #     print(q_module.fhe_circuit.mlir, file=f)

    # Test model in FHE mode
    if params.fhe_mode == 'simulate':
        use_sim = True
        loader = test_loader
    elif params.fhe_mode == 'execute':
        use_sim = False
        loader = execution_loader
        # Generate ciphertext keys
        t = time.time()
        q_module.fhe_circuit.keygen()
        print(f"Keygen time: {time.time() - t:.2f}s")

    # run validation set if testing accuracy of simulator
    # if use_sim:
    #     t = time.time()
    #     print(f"Running validation inference in {params.fhe_mode} mode...")
    #     top1_val, top5_val = test_with_concrete(
    #         params,
    #         q_module,
    #         val_loader,
    #         use_sim,
    #         cls=model.module.classifier
    #     )
    #     elapsed_time = time.time() - t
    #     time.sleep(5)
    #     time_per_inference = elapsed_time / params.test_subset
    #     print(
    #         f"Validation time per inference in FHE: {time_per_inference:.2f} | "
    #         f"Top-1 Acc: {top1_val.avg:.3f}% | Top-5 Acc: {top5_val.avg:.3f}%"
    #     )
    model.cuda()
    if use_sim:
        # Test on multiple random states
        random_states = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
        top1_plain = []
        top5_plain = []
        top1_enc = []
        top5_enc = []
        for rstate in random_states:
            print(f"Running validation inference on subset of {params.test_subset} with random state {rstate}...")
            if params.dataset == 'miniImagenet' and params.dct_status:
                test_loader, valset = base_datamgr_val.get_data_loader_dct(
                    base_file,
                    aug=False,
                    filter_size=params.filter_size,
                    subset=params.test_subset,
                    channels=params.channels,
                    random_state=rstate)
            elif params.dataset == 'miniImagenet' and not params.dct_status:
                test_loader, valset = base_datamgr_val.get_data_loader(
                    base_file,
                    aug=False,
                    subset=params.test_subset,
                    random_state=rstate)
            else:
                _, test_idx = train_test_split(np.arange(num_test), test_size=params.test_subset, random_state=rstate)
                test_sampler = SubsetRandomSampler(test_idx)

                test_loader = torch.utils.data.DataLoader(
                    testset,
                    batch_size=params.test_batch_size,
                    shuffle=False,
                    sampler=test_sampler,
                )

            # Test in plaintext
            model.cuda()
            top1_p, top5_p = test(model, criterion, test_loader, test_loader)
            top1_plain.append(top1_p.avg)
            top5_plain.append(top5_p.avg)

            # Test encrypted
            model.to(device)
            t = time.time()
            top1_val, top5_val = test_with_concrete(
                params,
                q_module,
                test_loader,
                use_sim,
                cls=model.module.classifier
            )
            elapsed_time = time.time() - t
            time.sleep(1)
            time_per_inference = elapsed_time / params.test_subset
            print(
                f"Test time per inference in FHE: {time_per_inference:.2f} | "
                f"Top-1 Acc: {top1_val.avg:.3f}% | Top-5 Acc: {top5_val.avg:.3f}% \n"
            )
            top1_enc.append(top1_val.avg)
            top5_enc.append(top5_val.avg)

        print(f'Plain top1 acc: {top1_plain}')
        print(f'Plain top5 acc: {top5_plain}')
        print(f'Encrypted top1 acc: {top1_enc}')
        print(f'Encrypted top5 acc: {top5_enc}')

    # t = time.time()
    # print(f"Running test inference in {params.fhe_mode} mode...")
    # top1_test, top5_test = test_with_concrete(
    #     params,
    #     q_module,
    #     test_loader,
    #     use_sim,
    #     cls=model.module.classifier
    # )
    # elapsed_time = time.time() - t
    # time.sleep(5)
    # time_per_inference = elapsed_time / params.test_subset
    # print(
    #     f"Test time per inference in FHE: {time_per_inference:.2f} | "
    #     f"Top-1 Acc: {top1_test.avg:.3f}% | Top-5 Acc: {top5_test.avg:.3f}%"
    # )
    print('Done')
    return


if __name__ == "__main__":
    try:
        main()
        os._exit(0)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)
    except SystemExit:
        os._exit(0)
