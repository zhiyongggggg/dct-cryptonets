import os
import gc
import numpy as np
import sys
import time
import torch
from torch.autograd import Variable
# from ../io_utils import model_dict
# from methods.baselinetrain import BaselineTrain
# from backbone import Conv4S, ResNet18, ResNet18dct, WideResNet16_8, ResNet18_QAT
import torchvision.transforms as transforms
# import datasets.cvtransforms as transforms_dct
import torch.backends.cudnn as cudnn
from thop import profile
from torchinfo import summary
from ptflops import get_model_complexity_info

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from io_utils import model_dict
from methods.baselinetrain import BaselineTrain
from models.backbone import Conv4S, ResNet18, ResNet18QAT
import data.cvtransforms as transforms_dct

filter_size = 8
device = 'cpu'

def measure(model, x, y):
    # synchronize gpu time and measure fp
    torch.cuda.synchronize()
    t0 = time.time()
    y_pred = model(x)
    torch.cuda.synchronize()
    elapsed_fp = time.time() - t0

    # zero gradients, synchronize time and measure
    model.zero_grad()
    t0 = time.time()
    y_pred.backward(y)
    torch.cuda.synchronize()
    elapsed_bp = time.time() - t0
    return elapsed_fp, elapsed_bp


def benchmark(model, x, y):
    # transfer the model on GPU
    model.cuda()

    # DRY RUNS
    for i in range(5):
        _, _ = measure(model, x, y)

    print('DONE WITH DRY RUNS, NOW BENCHMARKING')

    # START BENCHMARKING
    t_forward = []
    t_backward = []
    for i in range(10):
        t_fp, t_bp = measure(model, x, y)
        t_forward.append(t_fp)
        t_backward.append(t_bp)

    # free memory
    del model

    return t_forward, t_backward


def main():
    # set the seed for RNG
    if len(sys.argv) == 2:
        torch.manual_seed(int(sys.argv[1]))
    else:
        torch.manual_seed(1234)

    # set cudnn backend to benchmark config
    cudnn.benchmark = True

    # instantiate the models
    model_arg_dict = {
        'in_channels': 6,
        # 'pruning_percent': params.pruning,
        'img_size': 56,
    }
    # build the dict to iterate over
    architectures = {
        # 'Conv4S': Conv4S(),
        'ResNet18': ResNet18(**model_arg_dict),
        # 'ResNet18qat': ResNet18_QAT(),
        # 'ResNet18dct': ResNet18dct(),
         # 'WideResNet16_8': WideResNet16_8(),
    }

    # build dummy variables to input and output
    x = Variable(torch.randn(1, model_arg_dict['in_channels'], model_arg_dict['img_size'], model_arg_dict['img_size'])).to(device)
    y = torch.randn(1, 512).to(device)

    # loop over architectures and measure them
    for deep_net in architectures:
        print(deep_net)
        print(architectures[deep_net])
        summary(
            architectures[deep_net].to(device),
            input_size=(1, model_arg_dict['in_channels'], model_arg_dict['img_size'], model_arg_dict['img_size']),
            device=device
        )
        macs, params = profile(architectures[deep_net].to(device), inputs=(x,))
        print(f'MACs: {macs}, Params: {params}')

        if device == 'cuda':
            t_fp, t_bp = benchmark(architectures[deep_net], x, y)
            # print results
            print('FORWARD PASS: ', np.mean(np.asarray(t_fp) * 1e3), '+/-', np.std(np.asarray(t_fp) * 1e3))
            print('BACKWARD PASS: ', np.mean(np.asarray(t_bp) * 1e3), '+/-', np.std(np.asarray(t_bp) * 1e3))
            print('RATIO BP/FP:', np.mean(np.asarray(t_bp)) / np.mean(np.asarray(t_fp)))

        # with torch.cuda.device(0):
        #     macs, params = get_model_complexity_info(architectures[deep_net].cuda(), (6, 32, 32), as_strings=True,
        #                                              print_per_layer_stat=True, verbose=True)
        #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # # write the list of measures in files
        # fname = deep_net + '-benchmark.txt'
        # with open(fname, 'w') as f:
        #     for (fp_time, bp_time) in zip(t_fp, t_bp):
        #         f.write(str(fp_time) + " " + str(bp_time) + " \n")

        # force garbage collection
        gc.collect()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)
