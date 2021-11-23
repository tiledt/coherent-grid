import math
import torch
import models.networks as networks
import util.util as util
from data.base_dataset import TileImageToBatch
from scipy.ndimage.morphology import distance_transform_edt as DistTransform
from torch.nn.functional import interpolate
import torchvision.transforms.functional as F
import numpy as np
from kornia.losses.psnr import PSNRLoss
CPU = torch.device("cpu")
SPLITS = [4,16]

def resize(data, output_size=None, factor=None):
    if output_size is not None:
        return interpolate(data, size=output_size, mode="bilinear", align_corners=True)
    if factor is not None:
        return interpolate(data, scale_factor=factor, mode="bilinear", align_corners=True)

class Pix2PixTiledModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        self.tiles = SPLITS if opt.tiles == -1 else [opt.tiles]
        self.axis_splits = 1
        #axis_splits = int(math.sqrt(self.opt.tiles))
        #self.tile_adapter = TileImageToBatch(h_splits=axis_splits, w_splits=axis_splits)
        self.condition = opt.condition

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if opt.MSE_loss:
                self.criterionMSE = torch.nn.MSELoss()
            if opt.L1_loss:
                self.criterionL1 = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()
            if opt.use_weight_decay:
                self.WDLoss = torch.nn.MSELoss()
            if opt.PSNR_loss:
                self.criterionPSNR = PSNRLoss(1.)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            fake_image = self.do_inference(input_semantics)
            # with torch.no_grad():
            #     fake_image, _, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        print(netG)
        netD = networks.define_D(opt) if opt.isTrain else None
        #print(netD)
        netE = networks.define_E(opt) if opt.use_vae else None
        #print(netE)
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain and not opt.ignore_discriminator:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        if not(self.opt.no_one_hot):
            data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()

        # create one-hot label map
        if self.opt.no_one_hot:
            input_semantics = data['label']
        else:
            label_map = data['label']
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0)
        # concatenate instance map if it exists
        if not self.opt.no_instance_edge:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)
        if not self.opt.no_instance_dist:
            inst_map = data['instance']
            instance_dist = self.get_distransform(inst_map)
            input_semantics = torch.cat((input_semantics, instance_dist), dim=1)

        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image, lr_features, KLD_loss = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        if not self.opt.no_adv_loss:
            G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        if self.opt.MSE_loss:
            G_losses['MSE'] = self.criterionMSE(fake_image, real_image) \
                * self.opt.lambda_MSE
        if self.opt.PSNR_loss:
            G_losses["PSNR"] = self.criterionPSNR(fake_image, real_image) \
                * self.opt.lambda_PSNR
        if self.opt.L1_loss:
            G_losses['L1'] = self.criterionL1(fake_image, real_image) \
                * self.opt.lambda_L1
        if self.opt.use_weight_decay:
            lr_features_l2 = lr_features.norm(p=2)
            device = lr_features_l2.device
            zero = torch.zeros(lr_features_l2.shape).to(device)
            G_losses['WD'] = self.WDLoss(lr_features_l2, zero) \
                * self.opt.lambda_WD

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def do_inference(self, orig_input):

        device = orig_input.device
        orig_input = orig_input.half()
        
        tile = self.opt.tileSize
        h, w = orig_input.shape[-2:]

        # compute required padding
        right = (w % tile)
        bottom = (h % tile)
        if right != 0:
            right = tile - right
        if bottom != 0:
            bottom = tile - bottom

        w_splits = (w + right) // tile
        h_splits = (h + bottom) // tile


        if h == tile and w == tile:
            right = 0
            bottom = 0
            h_splits = 1
            w_splits = 1

        
        tile_batcher = TileImageToBatch(h_size=tile, w_size=tile)
        padded_input = F.pad(orig_input, [0, 0, right, bottom], padding_mode="reflect")
        tiled_input = tile_batcher.adapt(padded_input)
        small_padded_input = resize(padded_input.clone(), (tile, tile))

        with torch.no_grad():
            if self.condition == "preprocess":
                x_condition_small = self.netG(small_padded_input)
                if not isinstance(x_condition_small, list):
                    x_condition = resize(x_condition_small, padded_input.shape[-2:])
                    condition_tiles = tile_batcher.adapt(x_condition)
                else:
                    tb = TileImageToBatch(h_splits=h_splits, w_splits=w_splits)
                    condition_tiles = [tb.adapt(resize(cond, factor=(h_splits, w_splits))) for cond in x_condition_small]
            elif self.condition == "mixed":
                x_condition_small = self.netG(small_padded_input)
                x_condition = resize(x_condition_small, padded_input.shape[-2:])
                x_condition = torch.cat((x_condition, padded_input.clone()), dim=1)
                condition_tiles = tile_batcher.adapt(x_condition)
            elif self.condition == "same":
                condition_tiles = tiled_input.clone()
            else:
                condition_tiles = None
            
            sections = []

            for i in range(tiled_input.shape[0]):
                section = tiled_input[i:i+1,...]
                condition_section = None
                if condition_tiles is not None or self.condition== "none":
                    if isinstance(condition_tiles, list):
                        condition_section = [c[i:i+1,...] for c in condition_tiles]
                    elif self.condition == "none":
                        condition_section = [None] * 4
                    else:
                        condition_section = condition_tiles[i:i+1,...]
                section_hat = self.netG(section, condition_section)

                if torch.cuda.is_available:
                    torch.cuda.empty_cache()

                if isinstance(section_hat, tuple):
                    section_hat = section_hat[0]

                sections.append(section_hat)
        print(len(sections))
        fake_image = tile_batcher.reverse(sections, h_splits=h_splits, w_splits=w_splits)
        fake_image = fake_image[:, :, :h, :w] # remove padding

        return fake_image.float()

    def process_input(self, orig_input, input_tiles):

        if self.condition == "preprocess":
            # preprocess
            orig_input_small = resize(orig_input.clone(), input_tiles.shape[-2:])
            x_condition_small = self.netG(orig_input_small)
            if isinstance(x_condition_small, list):
                # sloppy but forces splits to be consistent
                x_condition = [resize(cond, factor=self.axis_splits) for cond in x_condition_small]
                condition_tiles = [self.tile_adapter.adapt(cond) for cond in x_condition]
            else:
                x_condition = resize(x_condition_small, orig_input.shape[-2:])
                condition_tiles = self.tile_adapter.adapt(x_condition)
            fake_tiled_image = self.netG(input_tiles, condition=condition_tiles)
        elif self.condition == "mixed":
            orig_input_small = resize(orig_input.clone(), input_tiles.shape[-2:])
            x_condition_small = self.netG(orig_input_small)
            x_condition = resize(x_condition_small, orig_input.shape[-2:])
            x_condition = torch.cat((x_condition, orig_input.clone()), dim=1)
            condition_tiles = self.tile_adapter.adapt(x_condition)
            fake_tiled_image = self.netG(input_tiles, condition=condition_tiles)
        elif self.condition == "same":
            condition_tiles = input_tiles.clone()
            fake_tiled_image = self.netG(input_tiles, condition=condition_tiles)
        elif self.condition == "none":
            condition_tiles = [None] * 4
            fake_tiled_image = self.netG(input_tiles, condition=condition_tiles)
        else:
            fake_tiled_image = self.netG(input_tiles)
        return fake_tiled_image


    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):

        self.axis_splits = int(math.sqrt(np.random.choice(self.tiles)))
        
        self.tile_adapter = TileImageToBatch(h_splits=self.axis_splits, w_splits=self.axis_splits)
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld
        # Given a batch of inputs, create a bigger batch with tiles, 
        # process it and reverse the tiling process to compose the final image
        tiled_input_semantics = self.tile_adapter.adapt(input_semantics)
        fake_tiled_image = self.process_input(input_semantics, tiled_input_semantics)
        #fake_tiled_image, lr_features = self.netG(tiled_input_semantics, z=z)
        fake_image = self.tile_adapter.reverse(fake_tiled_image)


        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        #return fake_image, lr_features, KLD_loss
        return fake_image, None, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        #edge = self.ByteTensor(t.size()).zero_()
        edge = self.ByteTensor(t.size()).zero_().bool()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def get_distransform(self, t):
        #edge = self.ByteTensor(t.size()).zero_().cpu()
        edge = self.ByteTensor(t.size()).zero_().bool().cpu()
        device = t.device
        t = t.cpu()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        distransform = DistTransform(edge)
        distransform = torch.from_numpy(distransform).float().to(device)
        return distransform.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
    
    def half(self):
        self.netG.half()
    
    def float(self):
        self.netG.float()
