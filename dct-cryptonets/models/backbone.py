import torch
import torch.nn as nn
import math
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat


def init_layer(L):
    """ Initialization using fan-in """
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class SimpleBlock(nn.Module):
    """ Simple ResNet Block """
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out


class SimpleQBlock(nn.Module):
    """ Quantized Simple ResNet Block """
    def __init__(self, indim, outdim, half_res, qconv_args, qidentity_args):
        super().__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = qnn.QuantConv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, **qconv_args)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = qnn.QuantConv2d(outdim, outdim, kernel_size=3, padding=1, **qconv_args)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = qnn.QuantReLU(bit_width=qidentity_args["bit_width"])
        self.relu2 = qnn.QuantReLU(bit_width=qidentity_args["bit_width"])
        self.quant_out = qnn.QuantIdentity(return_quant_tensor=False, scaling_init=1.0, **qidentity_args)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            self.shortcut = qnn.QuantConv2d(indim, outdim, 1, 2 if half_res else 1, **qconv_args)
            self.BNshortcut = nn.BatchNorm2d(outdim)
            self.BNquant_out = qnn.QuantIdentity(return_quant_tensor=False, scaling_init=1.0, **qidentity_args)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.quant_out(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNquant_out(self.BNshortcut(self.shortcut(x)))
        out = torch.add(out, short_out)
        out = self.relu2(out)
        return out


class ResNetDCT(nn.Module):
    """
    ResNet backbone for a range of traditional resnet perturbations.

    Args:
        block (typing.Callable[]): ResNet block function
        list_num_layers (list): list with number of blocks per channel
        list_out_dims (list): list of channel dimension per block
        flatten (bool): flatten last layer of encoder
        in_channels (int): image input channel dimension (for DCT / non-DCT methods)
        img_size (int): image input spatial dimension
        skip_single_downsample (bool): skip one of the downsampling blocks? (used for DCT-CryptoNets ResNet20)
    """
    def __init__(
            self,
            block,
            list_num_layers,
            list_out_dims,
            flatten=True,
            in_channels=24,
            img_size=224,
            skip_single_downsample=False,
    ):
        super().__init__()
        self.pruned_layers = set()
        net_perturbation = all_network_perturbations[f'{list_out_dims[0]}_{in_channels}_{img_size}']
        trunk = []

        if net_perturbation['conv1_kernel'] is not None:
            conv1 = nn.Conv2d(
                in_channels,
                list_out_dims[0],
                kernel_size=net_perturbation['conv1_kernel'],
                stride=net_perturbation['conv1_stride'],
                padding=net_perturbation['conv1_padding'],
                bias=False
            )
            bn1 = nn.BatchNorm2d(list_out_dims[0])
            trunk.extend([conv1, bn1])
            init_layer(conv1)
            init_layer(bn1)

        if 'relu1' not in net_perturbation.keys() or net_perturbation['relu1'] is not False:
            relu = nn.ReLU()
            trunk.extend([relu])

        if net_perturbation['pool1_kernel'] is not None:
            pool1 = nn.MaxPool2d(
                kernel_size=net_perturbation['pool1_kernel'],
                stride=net_perturbation['pool1_stride'],
                padding=1
            )
            trunk.extend([pool1])

        indim = list_out_dims[0]
        for i in range(len(list_num_layers)):
            for j in range(list_num_layers[i]):
                if skip_single_downsample:
                    half_res = (i >= 2) and (j == 0)
                else:
                    half_res = (i >= 1) and (j == 0)
                B = block(indim, list_out_dims[i], half_res)
                trunk.append(B)
                indim = list_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(net_perturbation['avgpool_kernel'])
            trunk.append(avgpool)
            trunk.append(nn.Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        return out


class ResNetQDCT(nn.Module):
    """
    Quantized ResNet backbone for a range of traditional resnet perturbations.

    Args:
        block (typing.Callable[]): ResNet block function
        list_num_layers (list): list with number of blocks per channel
        list_out_dims (list): list of channel dimension per block
        flatten (bool): flatten last layer of encoder
        in_channels (int): image input channel dimension (for DCT / non-DCT methods)
        img_size (int): image input spatial dimension
        bit_width (int): quantization bit-width (ideally 2^n)
        skip_single_downsample (bool): skip one of the downsampling blocks? (used for DCT-CryptoNets ResNet20)
    """
    def __init__(
            self,
            block,
            list_num_layers,
            list_out_dims,
            flatten=True,
            in_channels=24,
            img_size=224,
            bit_width=4,
            skip_single_downsample=False,
    ):
        super().__init__()
        net_perturbation = all_network_perturbations[f'{list_out_dims[0]}_{in_channels}_{img_size}']
        trunk = []

        # Quantization parameters
        self.qconv_args = {
            "weight_bit_width": bit_width,
            "weight_quant": Int8WeightPerTensorFloat,  # Is overwritten by weight_bit_width
            "bias": False,
            "bias_quant": None,
            "narrow_range": True
        }
        self.qidentity_args = {
            "bit_width": bit_width,
            "act_quant": Int8ActPerTensorFloat  # Is overwritten by bit_width
        }

        # Network layers
        if net_perturbation['conv1_kernel'] is not None:
            quant_inp = qnn.QuantIdentity(return_quant_tensor=False, **self.qidentity_args)
            conv1 = qnn.QuantConv2d(
                in_channels,
                list_out_dims[0],
                kernel_size=net_perturbation['conv1_kernel'],
                stride=net_perturbation['conv1_stride'],
                padding=net_perturbation['conv1_padding'],
                **self.qconv_args
            )
            bn1 = nn.BatchNorm2d(list_out_dims[0])
            trunk.extend([quant_inp, conv1, bn1])
            init_layer(conv1)
            init_layer(bn1)
        else:
            quant_inp = qnn.QuantIdentity(return_quant_tensor=False, **self.qidentity_args)
            trunk.extend([quant_inp])

        if 'relu1' not in net_perturbation.keys() or net_perturbation['relu1'] is not False:
            relu = qnn.QuantReLU(bit_width=self.qidentity_args["bit_width"])
            trunk.extend([relu])

        if net_perturbation['pool1_kernel'] is not None:
            pool1 = nn.MaxPool2d(
                kernel_size=net_perturbation['pool1_kernel'],
                stride=net_perturbation['pool1_stride'],
                padding=1
            )
            quant_out = qnn.QuantIdentity(return_quant_tensor=False, **self.qidentity_args)
            trunk.extend([pool1, quant_out])
        else:
            quant_out = qnn.QuantIdentity(return_quant_tensor=False, **self.qidentity_args)
            trunk.extend([quant_out])

        indim = list_out_dims[0]
        for i in range(len(list_num_layers)):
            for j in range(list_num_layers[i]):
                if skip_single_downsample:
                    half_res = (i >= 2) and (j == 0)
                else:
                    half_res = (i >= 1) and (j == 0)
                B = block(indim, list_out_dims[i], half_res, self.qconv_args, self.qidentity_args)
                trunk.append(B)
                indim = list_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(net_perturbation['avgpool_kernel'])
            trunk.append(avgpool)
            trunk.append(qnn.QuantIdentity(return_quant_tensor=False, **self.qidentity_args))
            trunk.append(nn.Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        return out


def ResNet20(flatten=True, in_channels=3, img_size=224):
    model = ResNetDCT(
        SimpleBlock,
        [3, 3, 3],
        [48, 56, 64],  # DCT-CrypoNets ResNet20 channel dimensions
        # [16, 32, 64],           # Traditional ResNet20 channel dimensions
        flatten,
        in_channels=in_channels,
        img_size=img_size,
        skip_single_downsample=True,  # Set to False if using Traditional ResNet20
    )
    return model


def ResNet20QAT(flatten=True, bit_width=4, in_channels=3, img_size=224):
    model = ResNetQDCT(
        SimpleQBlock,
        [3, 3, 3],
        [48, 56, 64],  # DCT-CrypoNets ResNet20 channel dimensions
        # [16, 32, 64],           # Traditional ResNet20 channel dimensions
        flatten,
        bit_width=bit_width,
        in_channels=in_channels,
        img_size=img_size,
        skip_single_downsample=True,  # Set to False if using Traditional ResNet20
    )
    return model


def ResNet18(flatten=True, in_channels=3, img_size=224):
    model = ResNetDCT(
        SimpleBlock,
        [2, 2, 2, 2],
        [64, 128, 256, 512],
        flatten,
        in_channels=in_channels,
        img_size=img_size,
    )
    return model


def ResNet18QAT(flatten=True, bit_width=4, in_channels=3, img_size=224):
    model = ResNetQDCT(
        SimpleQBlock,
        [2, 2, 2, 2],
        [64, 128, 256, 512],
        flatten,
        in_channels=in_channels,
        img_size=img_size,
        bit_width=bit_width
    )
    return model


# Network pertubations based on model architecture
# Key is "second (block) channel dimension" _ "input channel dimension" _ "input spatial dimension"
all_network_perturbations = {
    # Traditional ResNet20 models
    '16_3_32': {
        'conv1_kernel': 3,
        'conv1_stride': 1,
        'conv1_padding': 1,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 7,
    },
    # DCT-CryptoNets ResNet20 models
    '48_24_32': { # added for my own use
    'conv1_kernel': 1,
    'conv1_stride': 1,
    'conv1_padding': 0,
    'relu1': True,
    'pool1_kernel': None,
    'pool1_stride': None,
    'avgpool_kernel': 16,
    },
    '48_24_64': { # added for my own use
    'conv1_kernel': 1,
    'conv1_stride': 1,
    'conv1_padding': 0,
    'relu1': True,
    'pool1_kernel': None,
    'pool1_stride': None,
    'avgpool_kernel': 32,
    },
    '48_3_32': {
        'conv1_kernel': 3,
        'conv1_stride': 1,
        'conv1_padding': 1,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 7,
    },
    '48_24_8': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': True,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 3,
    },
    '48_24_16': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': True,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 7,
    },
    '48_48_8': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': True,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 3,
    },
    '48_48_16': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': True,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 7,
    },
    '64_48_16': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': True,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 3,
    },
    # RGB-Based ResNet18
    '64_6_32': { # added this for my own use case
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 3,
    },
    '64_3_32': {
        'conv1_kernel': 3,
        'conv1_stride': 1,
        'conv1_padding': 1,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 3,
    },
    '64_3_128': {
        'conv1_kernel': 7,
        'conv1_stride': 2,
        'conv1_padding': 3,
        'relu1': True,
        'pool1_kernel': 3,
        'pool1_stride': 2,
        'avgpool_kernel': 3,
    },
    '64_3_224': {
        'conv1_kernel': 7,
        'conv1_stride': 2,
        'conv1_padding': 3,
        'relu1': True,
        'pool1_kernel': 3,
        'pool1_stride': 2,
        'avgpool_kernel': 7,
    },
    '64_3_448': {
        'conv1_kernel': 7,
        'conv1_stride': 2,
        'conv1_padding': 3,
        'relu1': True,
        'pool1_kernel': 3,
        'pool1_stride': 2,
        'avgpool_kernel': 14,
    },
    '64_3_1024': {
        'conv1_kernel': 7,
        'conv1_stride': 2,
        'conv1_padding': 3,
        'pool1_kernel': 7,
        'pool1_stride': 4,
        'avgpool_kernel': 11,
    },
    # DCT-Based ResNet18
    '64_6_56': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 5,
    },
    '64_12_56': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 5,
    },
    '64_24_56': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 5,
    },
    '64_48_56': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 5,
    },
    '64_64_56': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 5,
    },
    '64_192_56': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 5,
    },
    '64_6_112': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 14,
    },
    '64_24_112': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 14,
    },
    '64_48_112': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 14,
    },
    '64_64_112': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 14,
    },
    '64_192_112': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 14,
    },
}
