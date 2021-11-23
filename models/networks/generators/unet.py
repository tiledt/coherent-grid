import torch
from torch import nn
from torch.nn import functional as F
from enum import Enum
from typing import Tuple
from math import floor
from models.networks.base_network import BaseNetwork

def activation(slope=None, inplace=False):
    if slope is not None:
        return nn.LeakyReLU(negative_slope=slope, inplace=inplace)
    return nn.ReLU(inplace=inplace)


def convolution(ic, oc, ks=3, stride=1, padding=0, bias=True, **kwargs):
    conv = nn.Conv2d(ic, oc, ks, stride, padding, bias=bias, **kwargs)
    return conv


def deconvolution(ic, oc, ks=3, stride=2, padding=1, bias=True):
    deconv = nn.ConvTranspose2d(ic, oc, ks, stride=stride, padding=padding, bias=bias, output_padding=1)
    return deconv


def norm(groups, channels):
    if groups is None:
        return nn.InstanceNorm2d(channels)
    return nn.GroupNorm(groups, channels)


class Upsample(nn.Module):
    def __init__(self, type: str = "bilinear"):
        super(Upsample, self).__init__()
        self.type = type

    def forward(self, x, output_size: Tuple[int] = None):
        scale_factor = None
        if output_size is None:
            # enforce scale factor if output_size is not provided
            scale_factor = 2.0
        if self.type == "bilinear" or self.type == "bicubic":
            return F.interpolate(x, size=output_size, scale_factor=scale_factor, mode=self.type, align_corners=True)
        return F.interpolate(x, size=output_size, scale_factor=scale_factor, mode='nearest')

    def __repr__(self):
        return f"Upsample(mode='{self.type}')"


class Downsample(nn.Module):
    def __init__(self, type: str = "bilinear"):
        super(Downsample, self).__init__()
        self.type = type

    def forward(self, x, output_size: Tuple[int] = None):
        scale_factor = None
        if output_size is None:
            # enforce scale factor if output_size is not provided
            scale_factor = .5
        if self.type == "bilinear" or self.type == "bicubic":
            return F.interpolate(x, size=output_size, scale_factor=scale_factor, mode=self.type, align_corners=True)
        return F.interpolate(x, size=output_size, scale_factor=scale_factor, mode='nearest')

    def __repr__(self):
        return f"Downsample(mode='{self.type}')"


class ProyectionBlock(nn.Module):
    """ProyectionBlock"""
    def __init__(self, in_channels, out_channels):
        super(ProyectionBlock, self).__init__()
        #self.conv = convolution(in_channels, out_channels, ks=7, padding=3, padding_mode='reflect')
        self.conv = convolution(in_channels, out_channels, ks=5, padding=2, padding_mode='reflect')

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """ResidualBlock Module"""
    def __init__(self, channels, slope=None, norm_groups=32, inplace=True):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            convolution(channels, channels, ks=3, padding=1, padding_mode='reflect', bias=False),
            norm(norm_groups, channels),
            activation(slope, inplace),
            nn.Dropout2d(.2),
            convolution(channels, channels, ks=3, padding=1, padding_mode='reflect', bias=False),
            norm(norm_groups, channels),
        )

    def forward(self, x):
        return x + self.layers(x)


class ResidualContractingBlock(nn.Module):
    """ResidualContractingBlock"""
    def __init__(self, in_channels, slope=None, norm_groups=32, expand_channels=True, inplace=True):
        super(ResidualContractingBlock, self).__init__()

        out_channels = in_channels * 2 if expand_channels else in_channels
        self.layers = nn.Sequential(
            convolution(in_channels, out_channels, ks=3, padding=1, stride=2, padding_mode='reflect', bias=False),
            norm(norm_groups, out_channels),
            activation(slope, inplace),
            nn.Dropout2d(.2),
            convolution(out_channels, out_channels, ks=3, padding=1, padding_mode='reflect', bias=False),
            norm(norm_groups, out_channels),
        )

        self.expand = nn.Sequential(
            convolution(in_channels, out_channels, ks=1, stride=2),
            norm(norm_groups, out_channels)
        )

        self.out_act = activation(slope, inplace)

    def forward(self, x):
        out = self.layers(x)
        identity = self.expand(x)
        out += identity
        out = self.out_act(out)
        return out


class ExpandingBlock(nn.Module):
    """ExpandingBlock with skip connection, it can be either dense or residual or noskip"""
    def __init__(self, in_channels,
                 slope=None,
                 norm_groups=32,
                 upsample_type: str = "nearest",
                 skip_type: str = "residual",
                 expand_channels=True,
                 inplace=True):

        super(ExpandingBlock, self).__init__()
        out_channels = in_channels // 2 if expand_channels else in_channels
        self.skip_type = skip_type
        self.up = Upsample(type=upsample_type)

        self.in_layers = nn.Sequential(
            convolution(in_channels, out_channels, ks=3, padding=1, padding_mode='reflect', bias=False),
            norm(norm_groups, out_channels),
            activation(slope, inplace)
        )

        if skip_type == "residual":
            self.skip_switch = nn.Parameter(torch.randn(1, out_channels, 1, 1))

        self.out_layers = nn.Sequential(
            convolution(out_channels * 2 if skip_type == "dense" else out_channels, out_channels,
                        ks=3, padding=1, padding_mode='reflect', bias=False),
            norm(norm_groups, out_channels),
            activation(slope, inplace))

    def forward(self, x, skip_x):
        output_size = skip_x.shape[-2:]
        x = self.up(x, output_size=output_size)

        x = self.in_layers(x)

        if self.skip_type == "noskip":
            xx = x
        elif self.skip_type == "residual":
            xx = x + (skip_x * self.skip_switch)
        else:
            xx = torch.cat((x, skip_x), dim=1)

        x = self.out_layers(xx)

        return x


class UnetGenerator(BaseNetwork):
    """UnetGenerator"""
    def __init__(self, opt = None, config: dict = None):

        super(UnetGenerator, self).__init__()


        if config is None:
            self.slope = opt.slope
            self.depth = opt.unet_depth
            self.base_channels = opt.base_channels
            self.norm_groups = opt.norm_groups
            self.res_blocks = opt.res_blocks
            self.in_channels =  opt.label_nc
            self.out_channels = opt.output_nc
            self.skip_type = opt.skip_type
            self.upsample_type = opt.upsample_type
        else:
            self.load_params(config["params"])

        self.in_features = ProyectionBlock(self.in_channels, self.base_channels)


        self.down_path = nn.ModuleList([
            ResidualContractingBlock(self.base_channels * (2 ** i), self.slope,
                                     self.norm_groups)
 
            for i in range(0, self.depth)
        ])

        self.residual = nn.Sequential(*[
            ResidualBlock(self.base_channels * (2 ** (self.depth)), self.slope,
                          self.norm_groups)
 
            for _ in range(self.res_blocks)
        ])

        self.up_path = nn.ModuleList([
            ExpandingBlock(self.base_channels * (2 ** i),
                           self.slope,
                           self.norm_groups,
                           #self.norm_groups * (2 ** i),
                           self.upsample_type,
                           skip_type=self.skip_type)
            for i in range(self.depth, 0, -1)
        ])


        self.out_features = ProyectionBlock(self.base_channels, self.out_channels)
 
        if config is not None:
            self.load_state_dict(config["state_dict"])


    def forward(self, x, _):
        x0 = self.in_features(x)

        skips = [x0]
        for i, module in enumerate(self.down_path):
            skips.append(module(skips[i]))

        xr = self.residual(skips.pop(-1))

        for i, module in enumerate(self.up_path):
            xr = module(xr, skips[-(i+1)])

        xout = self.out_features(xr)

        return xout
