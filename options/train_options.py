"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions
import argparse

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=5000, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=100, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tb_log', action='store_true', help='if specified, use tensorboard logging')

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--ignore_discriminator', action='store_true', help='Does not load the latest discriminator')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate. This is NOT the total #epochs. Total #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')

        # the default values for beta1 and beta2 differ by TTUR option
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)

        #parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--lambda_MSE', type=float, default=10.0, help='weight for MSE loss')
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        parser.add_argument('--lambda_WD', type=float, default=1e-8, help='weight WD loss')
        parser.add_argument('--lambda_PSNR', type=float, default=10.0, help='weight PSNR loss')
        parser.add_argument('--no_adv_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--MSE_loss', action='store_true', help='if specified, use MSE loss')
        parser.add_argument('--L1_loss', action='store_true', help='if specified, use L1 loss')
        parser.add_argument('--PSNR_loss', action='store_true', help='if specified, use PSNR loss')
        parser.add_argument('--use_weight_decay', action='store_true', help='if specified, use weight decay loss on the estimated parameters from LR')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--lambda_kld', type=float, default=0.05)
        parser.add_argument('--seed', type=int, default=77)
        parser.add_argument('--compute_PSNR', action='store_true')
        parser.add_argument('--not_deterministic', action='store_true', help="Disables deterministic")
        parser.add_argument('--use_amp', action='store_true', help="Enables automatic mixed precision")

        self.isTrain = True
        return parser


def create_validation_options(opt):
    val_opt = argparse.Namespace(**vars(opt))
    val_opt.preprocess_mode = "resize_crop"
    val_opt.crop_ratio = 0
    val_opt.shuffle = False
    val_opt.drop_last = True
    val_opt.max_dataset_size = 50
    val_opt.phase = "val"
    val_opt.batchSize = 2
    val_opt.load_size = 1024
    val_opt.crop_size = 512
    return val_opt
