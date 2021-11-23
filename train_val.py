"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions, create_validation_options
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from util.util import compute_psnrs, deterministic_behaviour
from trainers.pix2pix_trainer import Pix2PixTrainer
import torch


best_psnr = 0.0
# parse options
opt = TrainOptions().parse()

deterministic_behaviour(opt)
# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)
val_opt = create_validation_options(opt)
val_dataloader = data.create_dataloader(val_opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)
# log models
visualizer.log_models(trainer.pix2pix_model_on_one_gpu)

scaler = None
if opt.use_amp:
    scaler = torch.cuda.amp.GradScaler(enabled=True)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if (opt.D_steps_per_G == 0):
            trainer.run_generator_one_step(data_i, scaler)
        elif (i % opt.D_steps_per_G == 0):
            #start = time.time()
            trainer.run_generator_one_step(data_i, scaler)
            #torch.cuda.synchronize(device='cuda')
            #end = time.time()
            #f_time = end - start
            #print("time_%d:%f" % (i, f_time))

        # train discriminator

        if (opt.D_steps_per_G != 0):
            trainer.run_discriminator_one_step(data_i, scaler)

        if scaler is not None:
            scaler.update()

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses(opt.D_steps_per_G)
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,len(dataloader.dataset),
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)

    # Validation
    with torch.no_grad():
        trainer.set_eval()
        ## Beginning of validation
        losses = dict()
        psnrs = []
        for i, data_i in enumerate(val_dataloader):
            trainer.run_generator_one_step(data_i, scaler, phase="val")
            trainer.run_discriminator_one_step(data_i, scaler, phase="val")

            val_losses = trainer.get_latest_losses(opt.D_steps_per_G)

            for key, value in val_losses.items():
                if key not in losses:
                    losses[key] = []
                losses[key].append(value.item())

            if opt.compute_PSNR:
                current_psnrs = compute_psnrs(trainer.get_latest_generated(), data_i['image'], opt)
                psnrs += current_psnrs


        for key, value in losses.items():
            losses[key] = torch.Tensor(value)
        
        if opt.compute_PSNR:
            psnr = torch.Tensor(psnrs).mean()
            losses["val-PSNR"] = psnr
            if psnr >= best_psnr:
                best_psnr = psnr
                print(f"Saving best model so far at {epoch} with PSNR: {psnr}")
                trainer.save(f"best-psnr")

        visualizer.print_current_errors(epoch, iter_counter.epoch_iter, len(dataloader.dataset),losses, iter_counter.time_per_iter)
        visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        visuals = OrderedDict([('input_label', data_i['label']),
                                    ('synthesized_image', trainer.get_latest_generated()),
                                    ('real_image', data_i['image'])])

        visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far, phase="val")


    trainer.set_train()
    iter_counter.record_epoch_end()




    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
