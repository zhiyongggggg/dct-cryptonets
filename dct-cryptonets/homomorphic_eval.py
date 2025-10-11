"""
Homomorphic Evaluation of DCT-CryptoNets

author: Arjun Roy <roy208@purdue.edu>
"""

from __future__ import print_function

import os, sys
from tqdm import tqdm
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torchinfo import summary
from sklearn.model_selection import train_test_split
from concrete.ml.torch.compile import compile_torch_model, compile_brevitas_qat_model
from concrete.fhe import Configuration

# Local modules
from io_utils import model_dict, parse_args
from data.datamgr import SimpleDataManager
from utils import accuracy, AverageMeter, BaselineTrain


use_gpu = torch.cuda.is_available()
print(f'Using GPU: {use_gpu}\n')

cifar_path = "./cifardataset"


@torch.no_grad()
def test_unencrypted(model, criterion, data_loader):
    model.eval()
    with torch.no_grad():
        loss_avg = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            f, output = model.forward(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            loss_avg.update(loss.data.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

    return top1, top5, loss_avg


@torch.no_grad()
def test_encrypted(params, model, data_loader, fhe_mode, cls):
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    for data, target in tqdm(data_loader):
        data = data.numpy()

        # Quantize the inputs and cast to appropriate data type
        encoder_output = model.forward(data, fhe=fhe_mode)

        # Run through clear-text classifier
        try:
            output = cls.forward(torch.from_numpy(encoder_output).flatten().float())
        except:
            output = cls.forward(torch.from_numpy(encoder_output).float())

        # Measure accuracy and record loss
        if params.test_batch_size == 1:
            prec1, prec5 = accuracy(output.data.view(1, -1), target.data, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        top1.update(prec1.item(), data.shape[0])
        top5.update(prec5.item(), data.shape[0])

    return top1, top5


def main():
    # Initializations
    params = parse_args('homomorphic_eval')
    device = torch.device('cpu')

    # Check Quantization type
    if 'QAT' in str(params.model) or 'qat' in str(params.model):
        quantization_type = 'QAT'
    else:
        quantization_type = 'PTQ'

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
        test_datamgr = SimpleDataManager(
            params.image_size_dct,
            batch_size=params.test_batch_size,
            normalize_param=normalize_param,
            jitter_param=jitter_param,
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
            jitter_param=jitter_param,
        )
        test_transform = test_datamgr.trans_loader.get_composed_transform(
            aug=False
        )

    # Dataset
    if params.dataset == 'cifar10':
        trainset = datasets.CIFAR10(root=cifar_path, train=True, download=True, transform=test_transform)
        calibset = datasets.CIFAR10(root=cifar_path, train=True, download=True, transform=test_transform)
        valset = datasets.CIFAR10(root=cifar_path, train=True, download=True, transform=test_transform)
        testset = datasets.CIFAR10(root=cifar_path, train=False, download=True, transform=test_transform)

        num_train = len(trainset)
        train_idx, val_idx = train_test_split(np.arange(num_train), test_size=params.test_subset, random_state=42)
        val_sampler = SubsetRandomSampler(val_idx)

        num_test = len(testset)
        _, test_idx = train_test_split(np.arange(num_test), test_size=params.test_batch_size, random_state=42)
        test_sampler = SubsetRandomSampler(test_idx)

        calib_loader = torch.utils.data.DataLoader(calibset, batch_size=params.calib_batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=params.test_batch_size, sampler=val_sampler)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.test_batch_size, sampler=test_sampler)

    elif params.dataset == 'Imagenette':
        trainset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'train'), transform=test_transform)
        calibset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)
        valset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)
        testset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)

        num_train = len(trainset)
        train_idx, val_idx = train_test_split(np.arange(num_train), test_size=params.test_subset, random_state=42)
        val_sampler = SubsetRandomSampler(val_idx)

        num_test = len(testset)
        _, test_idx = train_test_split(np.arange(num_test), test_size=params.test_subset, random_state=42)
        test_sampler = SubsetRandomSampler(test_idx)

        calib_loader = torch.utils.data.DataLoader(calibset, batch_size=params.calib_batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=params.test_batch_size, sampler=val_sampler)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.test_batch_size, sampler=test_sampler)

    elif params.dataset == 'miniImagenet':
        base_file = params.dataset_path + 'base.json'
        test_file = params.dataset_path + 'val.json'
        if params.dct_status:
            base_datamgr_val = SimpleDataManager(params.image_size_dct, batch_size=params.test_batch_size)
            test_datamgr = SimpleDataManager(params.image_size_dct, batch_size=params.test_batch_size)

            calib_loader, valset = base_datamgr_val.get_data_loader_dct(
                base_file, aug=False, filter_size=params.filter_size, subset=params.test_subset, channels=params.channels)
            val_loader, valset = base_datamgr_val.get_data_loader_dct(
                base_file, aug=False, filter_size=params.filter_size, subset=params.test_subset, channels=params.channels)
            test_loader, testset = test_datamgr.get_data_loader_dct(
                test_file, aug=False, filter_size=params.filter_size, subset=params.test_subset, channels=params.channels)
        else:
            base_datamgr_val = SimpleDataManager(params.image_size, batch_size=params.test_batch_size)
            test_datamgr = SimpleDataManager(params.image_size, batch_size=params.test_batch_size)

            calib_loader, valset = base_datamgr_val.get_data_loader(base_file, aug=False, subset=params.test_subset)
            val_loader, valset = base_datamgr_val.get_data_loader(base_file, aug=False, subset=params.test_subset)
            test_loader, testset = test_datamgr.get_data_loader(test_file, aug=False, subset=params.test_subset)

    elif params.dataset == 'ImageNet':
        calibset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)
        valset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)
        testset = datasets.ImageFolder(root=os.path.join(params.dataset_path, 'val'), transform=test_transform)

        num_test = len(testset)
        _, test_idx = train_test_split(np.arange(num_test), test_size=params.test_subset, random_state=42)
        test_sampler = SubsetRandomSampler(test_idx)

        calib_loader = torch.utils.data.DataLoader(calibset, batch_size=params.calib_batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=params.test_batch_size, shuffle=False, sampler=test_sampler)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=params.test_batch_size, shuffle=False, sampler=test_sampler)

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

    """
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
    """

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint
    print('\nLoading checkpoint...')
    if params.checkpoint_path and os.path.exists(params.checkpoint_path):
        checkpoint = torch.load(params.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state'])
        model.module.best_prec1_val = checkpoint["prec1"]
        print(f'Loaded checkpoint {params.checkpoint_path} ({model.module.best_prec1_val:.3f}% Top-1 Acc. @ epoch {checkpoint["epoch"]})')
    else:
        print("WARNING: No checkpoint loaded. Using random weights (for testing only)")
        print("Results will NOT be meaningful!")

    # Create post-trained quantization calibration data which is first batch of train data
    for data, _ in calib_loader:
        calib_data = data.to(device)
        break

    # Create FHE model
    print(f'\nCompiling FHE Model (this can take up to 10 minutes for larger networks)...')
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
            # rounding_threshold_bits={"n_bits": params.rounding_threshold_bits, "method": "approximate"},  # makes things go brrr
            n_bits=params.n_bits,
            p_error=params.p_error,
            configuration=configuration,
            verbose=params.verbose,
        )
    elif quantization_type == 'PTQ':
        q_module = compile_torch_model(
            model.module.feature,
            calib_data,
            rounding_threshold_bits=params.rounding_threshold_bits,
            p_error=params.p_error,
            n_bits=params.n_bits,
            configuration=configuration,
            verbose=params.verbose,
        )
    elapsed_time = time.time() - t
    del calib_data
    print(f"Time for FHE compilation {elapsed_time:.2f}")

    # Check that the network is compatible with FHE constraints
    bitwidth = q_module.fhe_circuit.graph.maximum_integer_bit_width()
    print(
        f"Max bit-width: {bitwidth} bits" + " -> it works in FHE!!"
        if bitwidth <= 16
        else " too high for FHE computation"
    )

    # Generate MLIR
    if params.verbose:
        with open('mlir.txt', 'a') as f:
            print(q_module.fhe_circuit.mlir, file=f)

    # Generate ciphertext keys
    t = time.time()
    q_module.fhe_circuit.keygen()
    print(f"Keygen time: {time.time() - t:.2f}s")
    time.sleep(5)

    # Test model in (unencrypted) non-FHE mode
    print(f'\nRunning UNENCRYPTED model on a subset of {params.test_subset} images...')
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    top1_val, top5_val, loss_val = test_unencrypted(model, criterion, val_loader)
    top1_test, top5_test, loss_test = test_unencrypted(model, criterion, test_loader)
    print(f'[Validation] Top-1 Acc: {top1_val.avg:.3f}% | Top-5 Acc: {top5_val.avg:.3f}% | Avg. Loss: {loss_val.avg:.3f}')
    print(f'[Test] Top-1 Acc: {top1_test.avg:.3f}% | Top-5 Acc: {top5_test.avg:.3f}% | Avg. Loss: {loss_test.avg:.3f}')

    # Run validation set if testing accuracy of simulator
    model.to(device)
    if params.fhe_mode == 'simulate':
        t = time.time()
        print(f"\nRunning ENCRYPTED validation inference in {params.fhe_mode.upper()} mode on a subset of {params.test_subset} images...")
        top1_val, top5_val = test_encrypted(
            params,
            q_module,
            val_loader,
            params.fhe_mode,
            cls=model.module.classifier
        )
        elapsed_time = time.time() - t
        time.sleep(1)
        time_per_inference = elapsed_time / params.test_subset
        print(f'[Validation] Top-1 Acc: {top1_val.avg:.3f}% | Top-5 Acc: {top5_val.avg:.3f}% | '
              f'Time per inference in FHE: {time_per_inference:.2f}')

    # Run test set
    t = time.time()
    print(f"\nRunning ENCRYPTED test inference in {params.fhe_mode.upper()} mode on a subset of {params.test_subset} images...")
    top1_test, top5_test = test_encrypted(
        params,
        q_module,
        test_loader,
        params.fhe_mode,
        cls=model.module.classifier
    )
    elapsed_time = time.time() - t
    time.sleep(1)
    time_per_inference = elapsed_time / params.test_subset
    print(f'[Test] Top-1 Acc: {top1_test.avg:.3f}% | Top-5 Acc: {top5_test.avg:.3f}% | '
          f'Time per inference in FHE: {time_per_inference:.2f}')

    # Run reliability analysis over a range of 20 random datasets subsets
    if params.reliability_test is not None and params.fhe_mode == 'simulate':
        print('\n============ Encrypted Reliability Analysis ============')
        if torch.cuda.is_available():
            model.cuda()
        else:
            model.cpu()
        # Test on multiple random states
        random_states = [x for x in range(27, 29)]
        top1_plain = []
        top5_plain = []
        top1_enc = []
        top5_enc = []
        for rstate in random_states:
            print(f"\n\nRunning ENCRYPTED test inference on subset of {params.test_subset} with random state {rstate}...")
            if params.dataset == 'miniImagenet' and params.dct_status:
                test_loader, _ = base_datamgr_val.get_data_loader_dct(
                    base_file,
                    aug=False,
                    filter_size=params.filter_size,
                    subset=params.test_subset,
                    channels=params.channels,
                    random_state=rstate)
            elif params.dataset == 'miniImagenet' and not params.dct_status:
                test_loader, _ = base_datamgr_val.get_data_loader(
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

            # Plaintext accuracy on test-set
            if torch.cuda.is_available():
                model.cuda()
                print("Running on GPU")
            else:
                model.cpu()
                print("Running on CPU")
            top1_p, top5_p, loss_p = test_unencrypted(model, criterion, test_loader)
            top1_plain.append(top1_p.avg)
            top5_plain.append(top5_p.avg)
            print(f'[Test] UNENCRYPTED Top-1 Acc: {top1_p.avg:.3f}% | Top-5 Acc: {top5_p.avg:.3f}% | Avg. Loss: {loss_p.avg:.3f}')

            # Encrypted accuracy on test-set
            model.to(device)
            t = time.time()
            top1_e, top5_e = test_encrypted(
                params,
                q_module,
                test_loader,
                params.fhe_mode,
                cls=model.module.classifier
            )
            elapsed_time = time.time() - t
            time.sleep(1)
            time_per_inference = elapsed_time / params.test_subset
            print(f'[Test] ENCRYPTED Top-1 Acc: {top1_e.avg:.3f}% | Top-5 Acc: {top5_e.avg:.3f}% | '
                  f'Time per inference in FHE: {time_per_inference:.2f}')
            top1_enc.append(top1_e.avg)
            top5_enc.append(top5_e.avg)

        print(f'\n--------Encrypted Reliability Analysis Results--------')
        print(f'Unencrypted top1 acc: {top1_plain}')
        print(f'Unencrypted top5 acc: {top5_plain}')
        print(f'Encrypted top1 acc: {top1_enc}')
        print(f'Encrypted top5 acc: {top5_enc}')
        print(f'--------------------------------------------------------')

    print('Done')
    return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)
