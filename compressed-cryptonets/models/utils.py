import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Key is "second channel dimnsion" _ "input image dimension" _ "image spatial size"
# We use the "second channel dimension" to determine if we are using a cifar10 channel changes or imagenet one
all_network_perturbations = {
    '16_3_32': {
        'conv1_kernel': 3,
        'conv1_stride': 1,
        'conv1_padding': 1,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 7,
    },
    '64_3_32': {
        'conv1_kernel': 7,
        'conv1_stride': 1,
        'conv1_padding': 3,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 4,
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
    '64_6_16': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 2,
    },
    # '64_6_28': {
    #     'conv1_kernel': 7,
    #     'conv1_stride': 1,
    #     'conv1_padding': 3,
    #     'pool1_kernel': None,
    #     'pool1_stride': None,
    #     'avgpool_kernel': 3,
    # },
    # '64_24_28': {
    #     'conv1_kernel': 7,
    #     'conv1_stride': 1,
    #     'conv1_padding': 3,
    #     'pool1_kernel': None,
    #     'pool1_stride': None,
    #     'avgpool_kernel': 3,
    # },
    '64_6_32': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 3,
    },
    '64_24_32': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 3,
    },
    '64_48_32': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 3,
    },
    '64_64_32': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 3,
    },
    '64_192_32': {
        'conv1_kernel': 1,
        'conv1_stride': 1,
        'conv1_padding': 0,
        'relu1': False,
        'pool1_kernel': None,
        'pool1_stride': None,
        'avgpool_kernel': 3,
    },
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
        # 'conv1_kernel': 7,
        # 'conv1_stride': 1,
        # 'conv1_padding': 3,
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
        'relu1': True,
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
        # 'conv1_kernel': 7,
        # 'conv1_stride': 2,
        # 'conv1_padding': 3,
        'pool1_kernel': None,
        'pool1_stride': None,
        # 'pool1_kernel': 3,
        # 'pool1_stride': 2,
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


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def caffe2_xavier_init(module, bias=0):
    # `XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch
    # Acknowledgment to FAIR's internal code
    kaiming_init(
        module,
        a=1,
        mode='fan_in',
        nonlinearity='leaky_relu',
        distribution='uniform')


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class GumbleSoftmax(torch.nn.Module):
    def __init__(self, hard=False):
        super(GumbleSoftmax, self).__init__()
        self.hard = hard

    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        if self.gpu:
            return noise.cuda()
        else:
            return noise

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = logits.size(2)
        gumble_samples_tensor = self.sample_gumbel_like(logits.data)
        gumble_trick_log_prob_samples = logits + gumble_samples_tensor
        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, dim)
        return soft_samples

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probslibaba
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            # block layer
            # _, max_value_indexes = y.data.max(1, keepdim=True)
            # y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)
            # block channel
            _, max_value_indexes = y.data.max(2, keepdim=True)
            y_hard = logits.data.clone().zero_().scatter_(2, max_value_indexes, 1)
            y = Variable(y_hard - y.data) + y
        return y

    def forward(self, logits, temp=1, force_hard=False):
        samplesize = logits.size()

        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=1, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=1, hard=True)


class GateModule(nn.Module):
    def __init__(self, in_ch, kernel_size=28, doubleGate=False, dwLA=False):
        super(GateModule, self).__init__()

        self.doubleGate, self.dwLA = doubleGate, dwLA
        self.inp_gs = GumbleSoftmax()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_ch = in_ch

        if dwLA:
            if doubleGate:
                self.inp_att = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=1, padding=0, groups=in_ch,
                              bias=True),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.Sigmoid()
                )

            self.inp_gate = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=1, padding=0, groups=in_ch, bias=True),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(in_ch),
            )
            self.inp_gate_l = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, groups=in_ch,
                                        bias=True)
        else:
            if doubleGate:
                reduction = 4
                self.inp_att = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch // reduction, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(in_ch // reduction, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.Sigmoid()
                )

            self.inp_gate = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True),
            )
            self.inp_gate_l = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, groups=in_ch, bias=True)

    def forward(self, y, cb, cr, temperature=1.):
        if self.doubleGate:
            if self.dwLA:
                hatten_d1 = self.inp_att(x)
                hatten_d2 = self.inp_gate(x)
                hatten_d2 = self.inp_gate_l(hatten_d2)
            else:
                hatten_y, hatten_cb, hatten_cr = self.avg_pool(y), self.avg_pool(cb), self.avg_pool(cr)
                hatten = torch.cat((hatten_y, hatten_cb, hatten_cr), dim=1)

                hatten_d1 = self.inp_att(hatten)
                hatten_d2 = self.inp_gate(hatten)
                hatten_d2 = self.inp_gate_l(hatten_d2)

            hatten_d2 = hatten_d2.reshape(hatten_d2.size(0), self.in_ch, 2, 1)
            hatten_d2 = self.inp_gs(hatten_d2, temp=temperature, force_hard=True)
        else:
            if self.dwLA:
                hatten_d2 = self.inp_gate(x)
                hatten_d2 = self.inp_gate_l(hatten_d2)
            else:
                hatten_y, hatten_cb, hatten_cr = self.avg_pool(y), self.avg_pool(cb), self.avg_pool(cr)
                hatten_d2 = torch.cat((hatten_y, hatten_cb, hatten_cr), dim=1)
                hatten_d2 = self.inp_gate(hatten_d2)
                hatten_d2 = self.inp_gate_l(hatten_d2)

            hatten_d2 = hatten_d2.reshape(hatten_d2.size(0), self.in_ch, 2, 1)
            hatten_d2 = self.inp_gs(hatten_d2, temp=temperature, force_hard=True)

        if self.doubleGate:
            x = x * hatten_d1 * hatten_d2[:, :, 1].unsqueeze(2)
        else:
            y = y * hatten_d2[:, :64, 1].unsqueeze(2)
            cb = cb * hatten_d2[:, 64:128, 1].unsqueeze(2)
            cr = cr * hatten_d2[:, 128:, 1].unsqueeze(2)

        return y, cb, cr, hatten_d2[:, :, 1]


class GateModule192(nn.Module):
    def __init__(self, act='relu'):
        super(GateModule192, self).__init__()

        self.inp_gs = GumbleSoftmax()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_ch = in_ch = 192
        if act == 'relu':
            relu = nn.ReLU
        elif act == 'relu6':
            relu = nn.ReLU6
        else: raise NotImplementedError

        self.inp_gate = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_ch),
            relu(inplace=True),
        )
        self.inp_gate_l = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, groups=in_ch, bias=True)


    def forward(self, x, temperature=1.):
        hatten = self.avg_pool(x)
        hatten_d = self.inp_gate(hatten)
        hatten_d = self.inp_gate_l(hatten_d)
        hatten_d = hatten_d.reshape(hatten_d.size(0), self.in_ch, 2, 1)
        hatten_d = self.inp_gs(hatten_d, temp=temperature, force_hard=True)

        x = x * hatten_d[:, :, 1].unsqueeze(2)

        return x, hatten_d[:, :, 1]
