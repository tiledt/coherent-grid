import torch
from torch import nn
from torch.nn import functional as F
from enum import Enum
from typing import Tuple
from models.networks.base_network import BaseNetwork
import torch.nn.utils.spectral_norm as spectral_norm


def add_spectral(conv, opt):
    if "spectral" in opt.norm_G :
        return spectral_norm(conv)
    return conv


def activation(slope=None, inplace=False):
    if slope is not None:
        return nn.LeakyReLU(negative_slope=slope, inplace=inplace)
    return nn.ReLU(inplace=inplace)


def convolution(ic, oc, ks=3, stride=1, padding=0, bias=True, **kwargs):
    conv = nn.Conv2d(ic, oc, ks, stride, padding, bias=bias, **kwargs)
    return conv


def norm(groups, channels):
    if groups == -1:
        return nn.InstanceNorm2d(channels)
    elif groups == "batch":
        return nn.BatchNorm2d(channels)
    elif groups == "sync-batch":
        return nn.SyncBatchNorm(channels)
    return nn.GroupNorm(groups, channels)


class ConditionedNorm(nn.Module):
    def __init__(self, norm_groups, channels, opt,condition_channels=3):
        super(ConditionedNorm, self).__init__()
        self.interpolation = opt.interpolation
        self.norm = norm(norm_groups, channels)
        self.mlp = nn.Sequential(
            convolution(condition_channels, opt.condition_hidden_channels, ks=3, padding=1, padding_mode='reflect'),
            activation(opt.slope, False)
        )

        self.do_gamma = convolution(opt.condition_hidden_channels, channels, ks=3, padding=1, padding_mode='reflect')
        self.do_beta = convolution(opt.condition_hidden_channels, channels, ks=3, padding=1, padding_mode='reflect')

    def forward(self, x, condition):
        normalized = self.norm(x)
        if condition is None:
            return normalized
        if condition.size()[-2:] != x.size()[-2:]:
            if self.interpolation == "bilinear" or self.interpolation == "bicubic":
                condition = F.interpolate(condition, size=x.size()[-2:], mode=self.interpolation, align_corners=True)
            else:
                condition = F.interpolate(condition, size=x.size()[-2:], mode=self.interpolation)

        act = self.mlp(condition)
        gamma = self.do_gamma(act)
        beta = self.do_beta(act)

        # apply scale and bias
        normalized = normalized * (1 + gamma) + beta
        return normalized


class Upsample(nn.Module):
    def __init__(self, opt):
        super(Upsample, self).__init__()
        self.type = opt.interpolation

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
    def __init__(self, opt):
        super(Downsample, self).__init__()
        self.type = opt.interpolation

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
    def __init__(self, in_channels, out_channels, opt):
        super(ProyectionBlock, self).__init__()
        ks = opt.proyection_kernel
        self.layer = convolution(in_channels, out_channels, ks=ks, padding=ks // 2, padding_mode='reflect')
    def forward(self, x):
        x = self.layer(x)
        return x


class ResidualConditionedBlock(nn.Module):
    """ResidualConditionedBlock"""
    def __init__(self, in_channels, out_channels, condition_input_channels,  opt):

        super(ResidualConditionedBlock, self).__init__()

        norm_groups = out_channels // 2 if opt.norm_groups is None else opt.norm_groups

        self.norm_1 = ConditionedNorm(norm_groups, in_channels, opt, condition_input_channels)
        self.act_1 = activation(opt.slope, False)
        self.conv_1 = add_spectral(convolution(in_channels, in_channels, ks=3, padding=1, padding_mode='reflect', bias=False), opt)


        self.norm_2 = ConditionedNorm(norm_groups, in_channels, opt, condition_input_channels)
        self.act_2 = activation(opt.slope, False)
        self.conv_2 = add_spectral(convolution(in_channels, out_channels, ks=3, padding=1, padding_mode='reflect', bias=False), opt)


        self.norm_skip = ConditionedNorm(norm_groups, in_channels, opt, condition_input_channels)
        self.act_skip = activation(opt.slope, False)
        self.conv_skip = add_spectral(convolution(in_channels, out_channels, ks=3, padding=1, padding_mode='reflect', bias=False), opt)


    def forward(self, x, condition):
        x1 = self.conv_1(self.act_1(self.norm_1(x, condition)))
        x2 = self.conv_2(self.act_2(self.norm_2(x1, condition)))
        x_skip = self.conv_skip(self.act_skip(self.norm_skip(x, condition)))
        out = x2 + x_skip
        return out


class ResidualBlock(nn.Module):
    """ResidualBlock"""
    def __init__(self, in_channels, out_channels, opt):

        super(ResidualBlock, self).__init__()

        norm_groups = out_channels // 2 if opt.norm_groups is None else opt.norm_groups

        self.conv_1 = add_spectral(convolution(in_channels, in_channels, ks=3, padding=1, padding_mode='reflect', bias=False), opt)
        self.norm_1 = norm(norm_groups,in_channels)
        self.act_1 = activation(opt.slope, False)
        

        self.conv_2 = add_spectral(convolution(in_channels, out_channels, ks=3, padding=1, padding_mode='reflect', bias=False), opt)
        self.norm_2 = norm(norm_groups, out_channels)
        self.act_2 = activation(opt.slope, False)

        

        self.conv_skip = add_spectral(convolution(in_channels, out_channels, ks=3, padding=1, padding_mode='reflect', bias=False), opt)
        self.norm_skip = norm(norm_groups, out_channels)
        self.act_skip = activation(opt.slope, False)
       

    def forward(self, x):
        x1 = self.act_1(self.norm_1(self.conv_1(x)))
        x2 = self.act_2(self.norm_2(self.conv_2(x1)))
        x_skip = self.act_skip(self.norm_skip(self.conv_skip(x1)))
        out = x2 + x_skip
        return out


class MergeConditionedBlock(nn.Module):
    """MergeConditionedBlock"""
    def __init__(self, stream_channels, skip_channels, condition_input_channels, opt):

        super(MergeConditionedBlock, self).__init__()
        out_channels = stream_channels
        in_channels = stream_channels + skip_channels
        norm_groups = out_channels // 2 if opt.norm_groups is None else opt.norm_groups

        self.norm_1 = ConditionedNorm(norm_groups, in_channels, opt, condition_input_channels)
        self.act_1 = activation(opt.slope, False)
        self.conv_1 = add_spectral(convolution(in_channels, out_channels, ks=3, padding=1, padding_mode='reflect', bias=False), opt)

    def forward(self, x, skip, condition):
        x1 = self.conv_1(self.act_1(self.norm_1(torch.cat([x, skip], dim=1), condition)))
        return x1


class GlobalBlock(nn.Module):
    """GlobalBlock"""
    def __init__(self, opt):

        super(GlobalBlock, self).__init__()
        gb = opt.global_base_channels
        self.in_feat = ProyectionBlock(opt.label_nc, gb, opt)
        
        self.down = Downsample(opt)
        self.res0 = ResidualBlock(gb, 2 * gb, opt)
        self.res1 = ResidualBlock(2 * gb, 4 * gb, opt)
        self.res2 = ResidualBlock(4 * gb, 8 * gb, opt)
        self.res3 = ResidualBlock(8 * gb, 16 * gb, opt)
        

    def forward(self, x):
        feat = self.in_feat(x)
        #x0 = self.down(feat)
        x0 = self.res0(feat) # res: out 128

        x1 = self.down(x0) # res: out 64
        x1 = self.res1(x1) 

        x2 = self.down(x1) # res: out 32
        x2 = self.res2(x2) 

        x3 = self.down(x2) # res: out 16
        x3 = self.res3(x3) 

        return [x0, x1, x2, x3]


class ConditionedUnetGenerator(BaseNetwork):
    """ConditionedUnetGenerator"""
    def __init__(self, opt):
        super(ConditionedUnetGenerator, self).__init__()
        self.name = "ConditionedUnetV3Generator"

        self.padding_size = opt.padding_size

        self.pad = None if self.padding_size == 0 else nn.ReflectionPad2d(self.padding_size)
        self.global_stream = GlobalBlock(opt)

        bc = opt.base_channels
        gb = opt.global_base_channels
        df = opt.decoder_factor
        dc = bc * df
        self.in_feat = ProyectionBlock(opt.label_nc, bc, opt)
        
        self.down = Downsample(opt)
        self.up = Upsample(opt)

        self.res0 = ResidualBlock(    bc, 2 *  bc, opt)
        self.res1 = ResidualBlock(2 * bc, 4 *  bc, opt)
        self.res2 = ResidualBlock(4 * bc, 8 *  bc, opt)
        self.res3 = ResidualBlock(8 * bc, 16 * dc, opt)


        self.resu3 = ResidualBlock(16 * dc, 8 * dc, opt)
        self.merge3 = MergeConditionedBlock(8 * dc, 8 * bc, 16 * gb, opt)

        self.resu2 = ResidualBlock(8 * dc, 4 * dc, opt)
        self.merge2 = MergeConditionedBlock(4 * dc, 4 * bc, 8 * gb, opt)

        self.resu1 = ResidualBlock(4 * dc, 2 * dc, opt)
        self.merge1 = MergeConditionedBlock(2 * dc, 2 * bc, 4 * gb, opt)

        self.resu0 = ResidualBlock(2 * dc, dc, opt)
        self.merge0 = MergeConditionedBlock(dc, bc, 2 * gb, opt)

        self.out_feat = ProyectionBlock(dc, opt.output_nc, opt)

    def forward(self, x, condition=None):

        if condition is None:
            return self.global_stream(x)


        if self.pad is not None:
            x = self.pad(x)

        cond_u0, cond_u1, cond_u2, cond_u3 = condition

        # Feat
        feat_in = self.in_feat(x) # 128
        # Down
        xd0 = self.down(feat_in) # 64
        xd0 = self.res0(xd0)

        xd1 = self.down(xd0) # 32
        xd1 = self.res1(xd1) 

        xd2 = self.down(xd1) # 16
        xd2 = self.res2(xd2)

        xd3 = self.down(xd2) # 8
        xd3 = self.res3(xd3)
        

        # UP
        xu3 = self.resu3(xd3)
        xu3 = self.up(xu3, xd2.shape[-2:])
        xu3 = self.merge3(xu3, xd2, cond_u3)

        xu2 = self.resu2(xu3)
        xu2 = self.up(xu2, xd1.shape[-2:])
        xu2 = self.merge2(xu2, xd1, cond_u2)

        xu1 = self.resu1(xu2)
        xu1 = self.up(xu1, xd0.shape[-2:])
        xu1 = self.merge1(xu1, xd0, cond_u1)

        xu0 = self.resu0(xu1)
        xu0 = self.up(xu0, feat_in.shape[-2:])
        x_last = self.merge0(xu0, feat_in, cond_u0)
        
        # Feat
        feat_out = self.out_feat(x_last)

        if self.pad is not None:
            left = top = right = bottom = self.padding_size
            feat_out = feat_out[..., top:-bottom, left:-right]

        return feat_out



