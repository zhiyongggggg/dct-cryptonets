"""
Import(s) from:
https://github.com/tonylins/pytorch-mobilenet-v2
https://github.com/kaix90/DCTNet/classification/models/imagenet
"""
import torch
import torch.nn as nn
import math
from utils import *


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1., upscale=False):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        input_channel = 32
        last_channel = 1280
        if not upscale:
            self.cfgs = [
                # t, c, n, s
                [1,  16, 1, 1],
                [6,  24, 2, 2],
                [6,  32, 3, 1],
                [6,  64, 4, 2],
                [6,  96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        else:
            self.cfgs = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = make_divisible(input_channel * width_mult)
        # input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            layers.append(block(input_channel, output_channel, s, t))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# class MobileNetV2(nn.Module):
#     def __init__(self, n_class=1000, input_size=224, width_mult=1.):
#         super(MobileNetV2, self).__init__()
#         block = InvertedResidual
#         input_channel = 32
#         last_channel = 1280
#         interverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             [6, 160, 3, 2],
#             [6, 320, 1, 1],
#         ]
#
#         # building first layer
#         assert input_size % 32 == 0
#         # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
#         self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
#         self.features = [conv_bn(3, input_channel, 2)]
#         # building inverted residual blocks
#         for t, c, n, s in interverted_residual_setting:
#             output_channel = make_divisible(c * width_mult) if t > 1 else c
#             for i in range(n):
#                 if i == 0:
#                     self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
#                 else:
#                     self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         self.features.append(conv_1x1_bn(input_channel, self.last_channel))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*self.features)
#
#         # building classifier
#         self.classifier = nn.Linear(self.last_channel, n_class)
#
#         self._initialize_weights()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.mean(3).mean(2)
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()


class MobileNetV2DCT(nn.Module):
    def __init__(self, upscale_ratio=1, channels=0):
        super(MobileNetV2DCT, self).__init__()

        self.upscale_ratio = upscale_ratio
        in_ch, out_ch = channels, channels

        if upscale_ratio == 1:
            # model = mobilenetv2(pretrained=True, upscale=False)
            model = mobilenetv2(pretrained=False, upscale=False)
            self.deconv_cb = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True)
            )
            self.deconv_cr = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True)
            )
        elif upscale_ratio == 2:
            model = mobilenetv2(pretrained=True, upscale=True)
            self.deconv_y = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True)
            )
            self.deconv_cb = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True),
            )
            self.deconv_cr = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True)
            )

        self.input_layer = nn.Sequential(
            # pw
            nn.Conv2d(3*in_ch, 3*out_ch, 3, 1, 1, bias=False, groups=3*in_ch),
            nn.BatchNorm2d(3*out_ch),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(3*out_ch, 24, 1, 1, 0, bias=False),
            nn.BatchNorm2d(24),
        )

        self._initialize_weights()

        if upscale_ratio == 1:
            self.features = nn.Sequential(*list(model.children())[0][4:])
        elif upscale_ratio == 2:
            self.features = nn.Sequential(*list(model.children())[0][3:])
        self.conv = list(model.children())[1]
        self.avgpool = list(model.children())[2]
        self.classifier = list(model.children())[3]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, dct_y, dct_cb, dct_cr):
        if self.upscale_ratio == 1:
            dct_cb = self.deconv_cb(dct_cb)
            dct_cr = self.deconv_cr(dct_cr)
        elif self.upscale_ratio == 2:
            dct_y = self.deconv_y(dct_y)
            dct_cb = self.deconv_cb(dct_cb)
            dct_cr = self.deconv_cr(dct_cr)
        x = torch.cat((dct_y, dct_cb, dct_cr), dim=1)

        x = self.input_layer(x)
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MobileNetV2DCT_Subset_woinp(nn.Module):
    def __init__(self, channels=0, input_gate=False, num_classes=1000, input_size=224):
        super(MobileNetV2DCT_Subset_woinp, self).__init__()

        self.input_gate = input_gate

        # model = mobilenetv2(pretrained=True, upscale=True)
        model = mobilenetv2(pretrained=False, upscale=True, num_classes=num_classes)
        self.features = nn.Sequential(*list(model.children())[0][1:])
        if channels < 32:
            temp_layer = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
            temp_layer.weight.data = self.features[0].conv[0].weight.data[:channels]
            self.features[0].conv[0] = temp_layer

            temp_layer = nn.BatchNorm2d(channels)
            temp_layer.weight.data = self.features[0].conv[1].weight.data[:channels]
            temp_layer.bias.data = self.features[0].conv[1].bias.data[:channels]
            temp_layer.running_mean.data = self.features[0].conv[1].running_mean.data[:channels]
            temp_layer.running_var.data = self.features[0].conv[1].running_var.data[:channels]
            self.features[0].conv[1] = temp_layer

            temp_layer = nn.Conv2d(channels, self.features[0].conv[3].out_channels, 1, 1, 0, bias=False)
            temp_layer.weight.data = self.features[0].conv[3].weight.data[:, :channels]
            self.features[0].conv[3] = temp_layer
        elif channels == 192:
            out_ch = self.features[0].conv[0].out_channels
            temp_layer = nn.Conv2d(channels, out_ch, 3, 1, 1, groups=out_ch, bias=False)
            temp_layer.weight.data = self.features[0].conv[0].weight.data.repeat(1, 6, 1, 1)
            self.features[0].conv[0] = temp_layer

        self.conv = list(model.children())[1]
        self.avgpool = list(model.children())[2]
        self.classifier = list(model.children())[3]

        if input_gate:
            self.inp_GM = GateModule192()
            self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if 'inp_gate_l' in str(name):
                m.weight.data.normal_(0, 0.001)
                m.bias.data[::2].fill_(0.1)
                m.bias.data[1::2].fill_(2)
            elif 'inp_gate' in str(name):
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def forward(self, x):
        if self.input_gate:
            x, inp_atten = self.inp_GM(x)

        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.input_gate:
            return x, inp_atten
        else:
            return x


def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(width_mult=1, **kwargs)
    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        model.load_state_dict(state_dict)
    return model


def mobilenetv2dct(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2DCT(**kwargs)
    return model


def mobilenetv2dct_subset_woinp(**kwargs):
    """
    Constructs a DCT-based MobileNet V2 model
    """
    model = MobileNetV2DCT_Subset_woinp(**kwargs)
    return model
