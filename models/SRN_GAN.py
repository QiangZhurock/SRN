from .base_model import BaseModel
from .import networks
import torch
import itertools
from torch.autograd import Variable
from .vgg16 import Vgg16
import utils
import os
import torch.nn as nn

class SRN_GANModel(BaseModel):
    def name(self):
        return 'SRN_GANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.loss_names = ['D_B', 'G_B', 'identity_B', 'MSE_B']
        self.visual_names = ['real_Y', 'f', 'real_D']

        if self.isTrain:
            self.model_names = ['G_B', 'D_B']
        else:
            self.model_names = ['G_B']
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, not opt.norm,
                                        not opt.no_dropout, opt.init_type,
                                        self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm,
                                            use_sigmoid, opt.init_type,opt.gpu_ids)

        if self.isTrain:
            #define loss functions
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_Y = input['A']
        input_D = input['D']
        if len(self.gpu_ids) > 0:
            input_Y = input_Y.cuda(async = True)
            input_D = input_D.cuda(async = True)
        self.input_Y = input_Y
        self.input_D = input_D

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_Y = Variable(self.input_Y) #so that you can calculate grad
        self.real_D = Variable(self.input_D)

    def test(self):
        self.real_Y = Variable(self.input_Y, volatile = True) #so that you can calculate grad
        self.real_D = Variable(self.input_D)
        self.f = self.netG_B(self.real_Y)

    def backward_D_basic(self, netD, real, fake):
        #Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        #Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        #combine loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        #backward
        loss_D.backward()
        return loss_D

    def backward_D_B(self):
        f = self.f
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_D, f)

    def backward_G(self):
        lambda_G = max(self.opt.lambda_G * 0.995, 0.005)
        eta = max(self.opt.eta * 0.995, 0.0005)

        #GAN loss
        self.f = self.netG_B(self.real_Y)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.f), True)

        #MSE loss
        self.loss_MSE_B = self.criterionMSE(self.f, self.real_D)

        #identity loss between f and real_D
        vgg = Vgg16()
        utils.init_vgg16(self.opt.vgg_model_dir)
        vgg.load_state_dict(torch.load(os.path.join(self.opt.vgg_model_dir, "vgg16.weight")))
        if len(self.gpu_ids) > 0:
            vgg = vgg.cuda()

        real_D_copy = Variable(self.real_D.data.clone(),volatile = True)
        #vgg returns a list of four value
        features_f = vgg(self.f)
        features_D = vgg(real_D_copy)

        f_D_c = Variable(features_D[1].data, requires_grad = False)
        self.loss_identity_B = self.criterionMSE(features_f[1], f_D_c)

        #combine loss lambda_G = 0.01
        self.loss_G = self.loss_G_B * lambda_G + self.loss_MSE_B + self.loss_identity_B * 0.001
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        #D_B
        self.optimizer_D.zero_grad()
        self.backward_D_B()
        self.optimizer_D.step()





#change the architecture as my approack drawed