"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from models.pix2pix_tiled_model import Pix2PixTiledModel
from util.visualizer import Visualizer
from util import html
import time

import torch
import math

def test(opt):
    dataloader = data.create_dataloader(opt)

    check_gpu = torch.cuda.is_available() and len(opt.gpu_ids) > 0

    if opt.tiles == 0 and opt.tileSize is None:
        model = Pix2PixModel(opt)
    else:
        model = Pix2PixTiledModel(opt)
    model.eval()
    model.half()

    visualizer = Visualizer(opt)

    # create a webpage that summarizes the all results
    size = "full" if opt.preprocess_mode == "none" else f"crop-{opt.crop_size}"
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '%s_tile-%s_%s_%s' % (opt.phase, opt.tileSize, size, opt.which_epoch))
    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))
    time_total = 0
    # test
    for i, data_i in enumerate(dataloader):
        if check_gpu:
            torch.cuda.reset_max_memory_allocated()
        
        if i * opt.batchSize >= opt.how_many:
            break
        start = time.time()
        generated = model(data_i, mode='inference')
        if check_gpu:
            torch.cuda.synchronize(device='cuda')
        end = time.time()
        f_time = end-start
        if i != 0:
            time_total += f_time
        print("time_%d:%f"%(i,f_time))
        if check_gpu:
            print(torch.cuda.max_memory_allocated(device=None))

        img_path = data_i['path']
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('gt', data_i['image'][b]),
                                   ('synthesized_image', generated[b])])
            visualizer.save_images(webpage, visuals, img_path[b:b + 1], opt.image_format)

    webpage.save()
    print("average time per image = %f" % (time_total/(i)))

if __name__ == '__main__':
    opt = TestOptions().parse()
    # force test options
    opt.batchSize = 1
    opt.serial_batches = True
    # Fix these parameters
    #opt.load_size = None
    #opt.crop_size = 0 # 0 for not croping
    #opt.preprocess_mode = "none"
    test(opt)
