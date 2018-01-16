import torch
import torch.nn as nn
import torch.nn.functional as functional

from torch.autograd import Variable
from torch.nn.init import dirac, kaiming_normal
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable


def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def dirac_conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        DiracConv2d(in_dim, out_dim // 2, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(out_dim // 2),
        act_fn
    )
    return model


def conv_trans_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def dirac_conv_trans_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_3(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model


def dirac_conv_block_3(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        dirac_conv_block(in_dim, out_dim, act_fn),
        dirac_conv_block(out_dim, out_dim, act_fn),
        dirac_conv_block(out_dim, out_dim, act_fn),
    )
    return model


# dirac
def dirac_delta(ni, no, k):
    n = min(ni, no)
    size = (n, n) + k
    repeats = (max(no // ni, 1), max(ni // no, 1)) + (1,) * len(k)
    return dirac(torch.Tensor(*size)).repeat(*repeats)


def normalize(w):
    return functional.normalize(w.view(w.size(0), -1)).view_as(w)


class DiracConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, dilation=1, bias=True):
        super(DiracConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride=1, padding=padding, dilation=dilation, bias=bias)
        self.alpha = nn.Parameter(torch.Tensor([5]))
        self.beta = nn.Parameter(torch.Tensor([1e-5]))
        self.register_buffer('delta', dirac_delta(in_channels, out_channels, self.weight.size()[2:]))
        # print(in_channels)
        # print(self.delta.size())
        # print(self.weight.size())
        assert self.delta.size() == self.weight.size()

    def forward(self, input):
        return functional.conv2d(input, self.alpha * Variable(self.delta) + self.beta * normalize(self.weight),
                                 self.bias, self.stride, self.padding, self.dilation)


class NCRelu(nn.Module):
    def __init__(self, inplace=False):
        super(NCRelu, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return ncrelu(x)

    def __repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'


def ncrelu(x):
    return torch.cat([x.clamp(min=0), x.clamp(max=0)], dim=1)
