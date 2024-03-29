"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
#import wandb
import numpy as np
from torch.nn import L1Loss, MSELoss

NB_DATA = 7100

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    #opt.alpha = # A FAIRE
    opt.BPNN_Loss = MSELoss
   
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)    # regular setup: load and print networks; create schedulers
    
    dataset = create_dataset(opt,range(NB_DATA))  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    #wandb.init(entity='jhuboo', project=opt.name)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        psnr_metric = AverageMeter()    # initialize psnr 
        ssim_metric = AverageMeter()    # initialize ssim
        G_GAN_metric = AverageMeter()
        G_L1_metric = AverageMeter()
        D_fake_metric = AverageMeter()
        D_real_metric = AverageMeter()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
                
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
            # Metric evaluation and losses 
            ssim, psnr = model.metrics()
            psnr_metric.update(psnr.item(),opt.batch_size)
            ssim_metric.update(ssim.item(),opt.batch_size)
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                print("psnr  %f, ssim %f" %(psnr.item(),ssim.item()))
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            losses = model.get_current_losses()
            G_GAN_metric.update(losses["G_GAN"],opt.batch_size)
            G_L1_metric.update(losees["G_L1"],opt.batch_size)
            D_fake_metric.update(losses["D_fake"],opt.batch_size)
            D_real_metric.update(losses["D_real"],opt.batch_size)
                #metrics_dict = {"psnr": psnr_metric.avg,"ssim": ssim_metric.avg}
                #if opt.display_wandb is True:
                #    wandb.log(losses)
                #    wandb.log(metrics_dict)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                    print("suppose to display loss")

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        print("------ epoch %d ------" % (epoch))
        print("\n PSNR: %d, SSIM: %d \n" % (psnr_metric.avg, ssim_metric.avg))
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    
        #wandb.finish()
