import torch
from torchvision.utils import save_image
import torchvision.transforms as T
from .base_model import BaseModel
from . import networks
from .ssim import ssim
from torch.autograd import Variable
import cv2
from skimage.filters import threshold_otsu
import numpy as np
import os
import ntpath
from torch import nn
import torch.nn.functional as F

def bvtv_loss(tensor_to_count,tensor_mask):
    tensor_to_count = (tensor_to_count>0.5).type(torch.float32)
    ones = (tensor_to_count == 1).sum(dim=2)
    BV = ones.sum(dim=2)
    ones = (tensor_mask == 1).sum(dim=2)
    TV = ones.sum(dim=2)
    BVTV = BV/TV
    return BVTV

class ThrSigmoid(nn.Module):
    def __init__(self, k,t):
        super().__init__()
        self.k = k
        self.t = t
    def forward(self, x):
        ex = (1/(1+torch.exp(-self.k*(x-self.t))))
        return ex

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            #parser.add_argument('--lambda_BPNN', type=int, default=50.0, help='weight for BPNN loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to out. The training/test scripts will call <BaseModel.get_current_losses>
        if opt.BPNN_mode == "True":
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake','BPNN']
        else:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B','fake_B_bin','real_B_bin']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.deconv, opt.init_type, opt.init_gain, self.gpu_ids)
                                      
        # define a BPNN network for biological parameter estimation by Rehan
        self.BPNN_mode = opt.BPNN_mode
        if opt.BPNN_mode == "True":
            self.BPNN = networks.BPNN_model(features=64,out_channels=7,n1=158,n2=152,n3=83,k1=3,k2=3,k3=3)
            load_filename = "./checkpoints_bpnn/BPNN_checkpoint_lrhr.pth" # add by rehan
            self.alpha= opt.alpha
            #if isinstance(self.BPNN, torch.nn.DataParallel):
            #    self.BPNN = self.BPNN.module
            #print('loading the model from %s' % load_filename)
            #state_dict = torch.load(load_filename, map_location=str(self.device))
            #if hasattr(state_dict, '_metadata'):
            #    del state_dict._metadata

            # patch InstanceNorm checkpoints prior to 0.4
            #for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            #    self.__patch_instance_norm_state_dict(state_dict, self.BPNN, key.split('.'))
            #self.BPNN.load_state_dict(state_dict)
            self.BPNN.load_state_dict(torch.load(os.path.join(opt.checkpoint_BPNN,load_filename))) # load model parameters
            self.BPNN = self.BPNN.to(self.device)
            #if len(self.gpu_ids) > 0:
                #assert(torch.cuda.is_available())
                #self.BPNN.to(self.gpu_ids[0])
                #self.BPNN = torch.nn.DataParallel(self.BPNN, self.gpu_ids)  # multi-GPUs            
            print("---- BPNN mode ---- :", self.BPNN_mode)
            for p in self.BPNN.parameters():
                p.requires_grad = False
            self.BPNN.eval()
            self.sig = ThrSigmoid(k=400,t=0.21)
            self.gaussian_blur = T.GaussianBlur((5,5),3)
        
        print("number of gpu in pix2pix : ", self.gpu_ids)
        # define parameters for instance noise by Rehan
        #if self.isTrain:
            #self.inst_noise_sigma = opt.noise_sigma

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionBPNN = torch.nn.MSELoss()
            #self.criterionBPNN = opt.BPNN_Loss()
            #self.alpha = opt.alpha 
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.mask = input['mask'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        self.fake_B_bin = self.sig((self.gaussian_blur((((self.fake_B)+1)/2)*255)/255).clamp(0,1))        
    def Bio_param(self): # by Rehan
        """ Calculate biological parameters from fake image and corresponding real image"""
        
        gaussian_blur = T.GaussianBlur((5,5),3)
        real_B = self.real_B.clone()
        #real_B = gaussian_blur(real_B)
        #fake_B = fake_B.cpu().detach().numpy()
        real_B = real_B.cpu().detach()
        #real_B = ((real_B+1)/2)*255
        real_B = gaussian_blur(((real_B+1)/2)*255)
        real_B = real_B.numpy()/255>0.18
        #fake_B = fake_B.astype("float32")
        real_B = real_B.astype("float32")*1.
        #fake_B = np.ndarray(shape=,dtype
        #fake_B = torch.from_numpy(fake_B).to(self.device)
        real_B = torch.from_numpy(real_B).to(self.device)
        #save_image((real_B[0,:,:,:]+1)/2,"./real_B_bin_"+os.path.basename(self.image_paths[0])+".png")
        #save_image(self.fake_B_bin[0,:,:,:],"./fake_B_bin"+os.path.basename(self.image_paths[0])+".png")
        #save_image((self.real_A[0,:,:,:]+1)/2,"./real_A_"+os.path.basename(self.image_paths[0])+".png")
        self.real_B_bin = real_B
        masks_bin = self.mask.clone().detach()
        masks_bin = F.interpolate(masks_bin, size=64)
        self.P_fake = self.BPNN(masks_bin.to(self.device),self.fake_B_bin)
        self.P_real = self.BPNN(masks_bin.to(self.device),real_B)
        BVTV_SR = bvtv_loss(self.fake_B_bin,masks_bin.to(self.device))
        BVTV_HR = bvtv_loss(real_B,masks_bin.to(self.device))
        self.P_fake = torch.cat((self.P_fake,BVTV_SR),dim=1).clamp(-1,1)
        self.P_real = torch.cat((self.P_real,BVTV_HR),dim=1).clamp(-1,1)
        L1_BPNN = self.criterionBPNN(self.P_fake, self.P_real)
        return L1_BPNN
    
    def metrics(self):
        """Calculate PSNR and SSIM between fake_B and real_B.""" # Created by Rehan
        F_b = (Variable( self.fake_B, requires_grad=False) + 1) /2
        R_b = (Variable( self.real_B, requires_grad=False) + 1) /2
        short_path = ntpath.basename(self.image_paths[0])
        name = os.path.splitext(short_path)[0]
        with torch.no_grad():
            psnr_val = networks.PSNR(F_b, R_b, self.mask).cpu().detach().numpy()
            ssim_val = ssim(x=F_b, y=R_b, mask=self.mask, data_range=1., downsample= False).cpu().detach().numpy()
        return psnr_val, ssim_val
        
    #def Loss_extraction(self):
     #   BPNN_extraction = self.Bio_param() * self.opt.lambda_L1
      #  return BPNN_extraction

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        size_tensor = list(self.fake_B.size())
        #inst_noise_mean = torch.full((size_tensor[0], size_tensor[1], size_tensor[2], size_tensor[3]), 0, dtype = torch.float32, device=self.device)
        #inst_noise_std = torch.full((size_tensor[0], size_tensor[1], size_tensor[2], size_tensor[3]), self.inst_noise_sigma, dtype=torch.float32, device = self.device)
        #inst_noise = torch.normal(mean=inst_noise_mean,std=inst_noise_std).to(self.device) # Instance Noise added by Rehan
        #fake_AB = torch.cat((self.real_A, self.fake_B + inst_noise ), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator # Rehan adds inst_noise
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        #inst_noise = torch.normal(mean=inst_noise_mean,std=inst_noise_std).to(self.device) #Instance Noise added by Rehan
        #real_AB = torch.cat((self.real_A, self.real_B + inst_noise), 1) # Rehan adds inst_noise
        real_AB = torch.cat((self.real_A, self.real_B),1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        if self.BPNN_mode == "True":
            self.loss_BPNN = self.Bio_param()
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + (self.alpha * self.loss_BPNN)
        else :
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
