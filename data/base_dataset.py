"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from typing import Union, Tuple
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional_pil import resize
from util import util

def normalization_values(selection):
    if selection == "default":
        return torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5])
    elif selection == "imagenet":
        return torch.Tensor([0.485, 0.456, 0.406]), torch.Tensor([0.229, 0.224, 0.225])

def pil_transform_from_torch(torch_transform):
    return {
        InterpolationMode.NEAREST: Image.NEAREST,
        InterpolationMode.BILINEAR: Image.BILINEAR,
        InterpolationMode.BICUBIC: Image.BICUBIC,
    }[torch_transform]


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess_mode == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess_mode == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h / w
    elif opt.preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(opt.load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    if opt.center_crop:
        x = (opt.load_size-opt.crop_size)//2
        y = (opt.load_size*(h/w)-opt.crop_size)//2
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=InterpolationMode.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    if 'resize' in opt.preprocess_mode:
        osize = [opt.load_size, opt.load_size]
        #transform_list.append(transforms.Resize(osize, interpolation=method))
        transform_list.append(transforms.Lambda(lambda img: _resize(img, osize, method)))
    elif 'scale_width' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
    elif 'scale_shortside' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, method)))

    if 'crop' in opt.preprocess_mode:
        if torch.rand(1).item() <= opt.crop_ratio:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
        else:
            #transform_list.append(transforms.Resize(opt.crop_size, interpolation=method))
            transform_list.append(transforms.Lambda(lambda img: _resize(img, (opt.crop_size,opt.crop_size), method)))

    if opt.preprocess_mode == 'none':
        base = 32
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.preprocess_mode == 'fixed':
        w = opt.crop_size
        h = round(opt.crop_size / opt.aspect_ratio)
        transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        if opt.imagenet_norm:
            transform_list += [util.Normalize()]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transform_list


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __resize(img, w, h, method=InterpolationMode.BICUBIC):
    return img.resize((w, h), pil_transform_from_torch(method))


def _resize(img, size, method=InterpolationMode.BICUBIC):
    if img.mode == "RGB":
        return resize(img, size, pil_transform_from_torch(method))
    # For
    return resize(img, size, pil_transform_from_torch(InterpolationMode.NEAREST))


def __make_power_2(img, base, method=InterpolationMode.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), pil_transform_from_torch(method))


def __scale_width(img, target_width, method=InterpolationMode.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), pil_transform_from_torch(method))


def __scale_shortside(img, target_width, method=InterpolationMode.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), pil_transform_from_torch(method))


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))



def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


class TileImageToBatch:
    """
    Generates a tiled batch from a batch of images
    """
    def __init__(self,
                 initial_batch_size=1,
                 h_splits: int = None,
                 w_splits: int = None,
                 h_size: int = None,
                 w_size: int = None):

        if h_splits is None and h_size is None:
            raise ValueError("You have to provide splits or size")

        self.initial_batch_size = initial_batch_size
        self.h_splits = h_splits
        self.w_splits = w_splits
        self.h_size = h_size
        self.w_size = w_size

    def __str__(self):
        return f"initial_batch_size(layers={self.initial_batch_size}, h_splits={self.h_splits}, w_splits={self.w_splits}, h_size={self.h_size}, w_size={self.w_size})"
    
    def __repr__(self):
        return self.__str__()

    def do_tiles(self, data, h_size, w_size, h_splits, w_splits):

        if self.initial_batch_size != data.shape[0]:
            # override batch size
            self.initial_batch_size = data.shape[0]

        data = data.permute((2, 3, 1, 0))  # to H, W, C, B
        batch = []
        for i in range(self.initial_batch_size):
            parts = []
            for h in range(h_splits):
                w_parts = []
                for w in range(w_splits):
                    h_start = h * h_size
                    h_end = h_start + h_size
                    w_start = w * w_size
                    w_end = w_start + w_size
                    part = data[h_start:h_end, w_start:w_end, :, i]
                    # iteratively concat parts by last dimension, the batch dimension
                    w_parts.append(part.unsqueeze(3))
                parts.append(torch.cat(w_parts, dim=3))
            batch.append(torch.cat(parts, dim=3))
        # make a big batch out of the individual ones
        batch = torch.cat(batch, dim=-1)
        # permute it bach to B,C,H,W
        data = batch.permute((3, 2, 0, 1))
        return data

    def adapt(self, x) -> torch.Tensor:
        # from here x and y should have a shape like B, C, H, W
        h_splits = self.h_splits
        w_splits = self.w_splits
        h_size = self.h_size
        w_size = self.w_size
        H, W = x.shape[-2:]

        if h_splits is None:  # assumes w_splits is also None
            if H % h_size != 0 and W % w_size != 0:
                raise ValueError(f"({H},{W}) incompatible with sizes ({h_size},{w_size})")
            h_splits = H // h_size
            w_splits = W // w_size
        elif h_size is None:  # assumes w_size is also None
            if H % h_splits != 0 and W % w_splits != 0:
                raise ValueError(f"({H},{W}) incompatible with splits ({h_splits},{w_splits})")
            h_size = H // h_splits
            w_size = W // w_splits

        x_tile = self.do_tiles(x, h_size, w_size, h_splits, w_splits)
        return x_tile

    def reverse(self, data: Union[torch.Tensor, list], h_splits: int = None, w_splits: int = None) -> torch.Tensor:
        if type(data) == list:
            data = torch.cat(data, dim=0)

        if h_splits is None and self.h_splits is None:  # assumes square root
            size = data.shape[0]
            h_splits, w_splits = int(size ** .5), int(size ** .5)
        elif self.h_splits is not None:
            h_splits, w_splits = self.h_splits, self.w_splits

        batch = []
        for i in range(self.initial_batch_size):
            mosaic = []
            for h in range(h_splits):
                w_parts = []
                for w in range(w_splits):
                    part = (i * h_splits * w_splits) + (h * w_splits) + w
                    w_parts.append(data[part, :, :, :].unsqueeze(0))

                mosaic.append(torch.cat(w_parts, dim=-1))
            batch.append(torch.cat(mosaic, dim=-2))

        batch = torch.cat(batch, dim=0)
        return batch