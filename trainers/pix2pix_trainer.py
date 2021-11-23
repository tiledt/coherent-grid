"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
from models.pix2pix_tiled_model import Pix2PixTiledModel


class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        if opt.tiles == 0:
            print("Using Regular Trainer")
            self.pix2pix_model = Pix2PixModel(opt)
        else:
            print("Using Tiled Trainer")
            self.pix2pix_model = Pix2PixTiledModel(opt)

        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    # def run_generator_one_step(self, data):
    #     self.optimizer_G.zero_grad()
    #     g_losses, generated = self.pix2pix_model(data, mode='generator')
    #     g_loss = sum(g_losses.values()).mean()
    #     g_loss.backward()
    #     self.optimizer_G.step()
    #     self.g_losses = g_losses
    #     self.generated = generated

    def run_generator_one_step(self, data, scaler=None, phase="train"):
        if phase == "train":
            self.optimizer_G.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                g_losses, generated = self.pix2pix_model(data, mode='generator')
        else:
            g_losses, generated = self.pix2pix_model(data, mode='generator')

        g_loss = sum(g_losses.values()).mean()

        if phase == "train":
            if scaler is not None:
                scaler.scale(g_loss).backward()
                scaler.step(self.optimizer_G)
            else:
                g_loss.backward()
                self.optimizer_G.step()
        else:
            g_losses = {f"{phase}-{key}":value  for key, value in g_losses.items()}

        self.g_losses = g_losses
        self.generated = generated

    # def run_discriminator_one_step(self, data):
    #     self.optimizer_D.zero_grad()
    #     d_losses = self.pix2pix_model(data, mode='discriminator')
    #     d_loss = sum(d_losses.values()).mean()
    #     d_loss.backward()
    #     self.optimizer_D.step()
    #     self.d_losses = d_losses

    def run_discriminator_one_step(self, data, scaler=None, phase="train"):
        if phase == "train":
            self.optimizer_D.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                d_losses = self.pix2pix_model(data, mode='discriminator')
        else:
                d_losses = self.pix2pix_model(data, mode='discriminator')
        
        d_loss = sum(d_losses.values()).mean()
        
        if phase == "train":
            if scaler is not None:
                scaler.scale(d_loss).backward()
                scaler.step(self.optimizer_D)
            else:
                d_loss.backward()
                self.optimizer_D.step()
        else:
            d_losses = {f"{phase}-{key}":value  for key, value in d_losses.items()}

        self.d_losses = d_losses



    # def run_generator_one_step_val(self, data):

    #     g_losses, generated = self.pix2pix_model(data, mode='generator')
    #     g_loss = sum(g_losses.values()).mean()

    #     self.g_losses = {f"val-{key}":value  for key, value in g_losses.items()}
    #     self.generated = generated

    # def run_discriminator_one_step_val(self, data):
        
    #     d_losses = self.pix2pix_model(data, mode='discriminator')
    #     d_loss = sum(d_losses.values()).mean()
        
    #     self.d_losses = {f"val-{key}":value  for key, value in d_losses.items()}


    def get_latest_losses(self, D_steps_per_G):
        if (D_steps_per_G != 0):
            return {**self.g_losses, **self.d_losses}
        else:
            return {**self.g_losses}


    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def set_eval(self):
        for net in [self.pix2pix_model_on_one_gpu.netD, self.pix2pix_model_on_one_gpu.netG, self.pix2pix_model_on_one_gpu.netE]:
            if net is not None:
                net.eval()
        
    
    def set_train(self):
        for net in [self.pix2pix_model_on_one_gpu.netD, self.pix2pix_model_on_one_gpu.netG, self.pix2pix_model_on_one_gpu.netE]:
            if net is not None:
                net.train()
