""" PolyKervNets and Poly-Net backbone """

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import conv2d
import torch.nn.functional as F
import numpy as np


def actwrapper(module, act_fn=None):
    for name, layer in module._modules.items():
        actwrapper(layer, act_fn)

        if act_fn is not None and isinstance(layer, nn.ReLU):
            # Replace ReLU layer with the specified activation function
            act = act_fn()
            module._modules[name] = act

        if act_fn is not None and isinstance(layer, nn.MaxPool2d):
            # Create replacement AvgPool2D layer with the same kernel size and stride
            avg_pool = nn.AvgPool2d(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
            )
            module._modules[name] = avg_pool


def convwrapper(module, kernel=None, device='cuda'):
    for name, layer in module._modules.items():
        convwrapper(layer, kernel, device)

        if kernel is not None and isinstance(layer, nn.Conv2d):
            conv1 = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, kernel,
                        layer.stride, layer.padding, layer.dilation, layer.groups, layer.bias,
                        layer.padding_mode)
            module._modules[name] = conv1**2

            # pkn = PKN2d(layer.in_channels, layer.out_channels, layer.kernel_size, kernel,
            #             layer.stride, layer.padding, layer.dilation, layer.groups, layer.bias,
            #             layer.padding_mode, device)
            # module._modules[name] = pkn

        if kernel is not None and isinstance(layer, nn.ReLU):
            # Replace ReLU layer with the specified activation function
            act = nn.Identity()
            module._modules[name] = act

        if kernel is not None and isinstance(layer, nn.MaxPool2d):
            # Create replacement AvgPool2D layer with the same kernel size and stride
            avg_pool = nn.AvgPool2d(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
            )
            module._modules[name] = avg_pool

        # Needed to compile concrete-ml circuit
        if kernel is not None and isinstance(layer, nn.AdaptiveAvgPool2d):
            avg_pool = nn.AvgPool2d(
                kernel_size=7,
                stride=7,
                padding=0,
            )
            module._modules[name] = avg_pool


class ActPKN(torch.nn.Module):
    def __init__(self, cp=1.0, dp=2, dropout=False, learnable=True):
        super(ActPKN, self).__init__()
        self.cp = torch.nn.parameter.Parameter(torch.tensor(cp, requires_grad=learnable))
        self.dp = dp
        self.dropout = dropout

    def forward(self, x):
        out = (x + self.cp) ** self.dp
        if self.dropout:
            out = torch.nn.Dropout()(out)
        return out


class ReactPKN(torch.nn.Module):
    def __init__(self, ap=0.009, bp=0.5, cp=0.47, dp=2, dropout=False, learnable=True):
        super(ReactPKN, self).__init__()
        self.ap = torch.nn.parameter.Parameter(torch.tensor(ap, requires_grad=learnable))
        self.bp = torch.nn.parameter.Parameter(torch.tensor(bp, requires_grad=learnable))
        self.cp = torch.nn.parameter.Parameter(torch.tensor(cp, requires_grad=learnable))
        self.dp = dp
        self.dropout = dropout

    def forward(self, x):
        out = (self.ap * (x ** self.dp)) + (self.bp * x) + self.cp
        if self.dropout:
            out = torch.nn.Dropout()(out)
        return out


class LinearKernel(torch.nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def forward(self, x_unf, w, b):
        t = x_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        if b is not None:
            return t + b
        return t


class PKN(LinearKernel):
    def __init__(self, cp=1.0, dp=2, learnable=True):
        super(PKN, self).__init__()
        self.cp = torch.nn.parameter.Parameter(torch.tensor(cp, requires_grad=learnable))
        self.dp = dp

    def forward(self, x_unf, w, b):
        return (self.cp + super(PKN, self).forward(x_unf, w, b)) ** self.dp


class RPKN(LinearKernel):
    def __init__(self, ap=0.009, cp=0.47, dp=2, learnable=True):
        super(RPKN, self).__init__()
        self.ap = torch.nn.parameter.Parameter(torch.tensor(ap, requires_grad=learnable))
        self.cp = torch.nn.parameter.Parameter(torch.tensor(cp, requires_grad=learnable))
        self.dp = dp

    def forward(self, x_unf, w, b):
        # print(((self.cp + super(RPKN, self).forward(x_unf, w, b))**self.dp)*self.ap)
        # print(torch.max(((self.cp + super(RPKN, self).forward(x_unf, w, b))**self.dp)*self.ap))
        # print(torch.min(((self.cp + super(RPKN, self).forward(x_unf, w, b))**self.dp)*self.ap))
        return ((self.cp + super(RPKN, self).forward(x_unf, w, b)) ** self.dp) * self.ap


class PKN2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_fn=RPKN,
                 stride=1, padding=0, dilation=1, groups=1, bias=None,
                 padding_mode='zeros', device='cuda'):
        '''
        Follows the same API as torch Conv2d except kernel_fn.
        kernel_fn should be an instance of the above kernels.
        '''
        super(PKN2d, self).__init__(in_channels, out_channels,
                                    kernel_size, stride, padding,
                                    dilation, groups, bias, padding_mode,
                                    device)
        self.device = device
        self.kernel_fn = RPKN()

    def custom_unfold(self, input_tensor, kernel_size, dilation=1, padding=0, stride=1):
        batch_size, channels, height, width = input_tensor.size()

        # Apply padding if necessary
        if padding > 0:
            input_tensor = torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding))

        # Compute unfolded height and width
        unfolded_height = (height + 2 * padding - dilation * (kernel_size[0] - 1) - 1) // stride + 1
        unfolded_width = (width + 2 * padding - dilation * (kernel_size[1] - 1) - 1) // stride + 1

        # Initialize empty tensor for unfolded output
        unfolded_tensor = torch.empty(batch_size, channels * kernel_size[0] * kernel_size[1],
                                      unfolded_height * unfolded_width, dtype=input_tensor.dtype,
                                      device=self.device)

        # Iterate through each position in the unfolded output and fill it
        for i in range(unfolded_height):
            for j in range(unfolded_width):
                # Compute starting position of the patch
                h_start = i * stride
                w_start = j * stride

                # Extract the patch from the input tensor
                patch = input_tensor[:, :,
                        h_start:(h_start + kernel_size[0] * dilation),
                        w_start:(w_start + kernel_size[1] * dilation)]

                # Reshape and store the patch in the unfolded tensor
                unfolded_tensor[:, :, i * unfolded_width + j] = patch.contiguous().view(batch_size, -1)

        return unfolded_tensor

    def compute_shape(self, x):
        h = (x.shape[2] + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w = (x.shape[3] + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        return h, w

    def forward(self, x):
        # x_unf = self.custom_unfold(x, self.kernel_size, self.dilation[0], self.padding[0], self.stride[0])
        x_unf = torch.nn.functional.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)
        h, w = self.compute_shape(x)
        return self.kernel_fn(x_unf, self.weight.to(self.device), self.bias).view(x.shape[0], -1, h, w)
