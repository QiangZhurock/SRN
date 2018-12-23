from .base_model import BaseModel
from .import networks
import torch
import itertools
from torch.autograd import Variable

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
                                        not opt.drop_out, opt.init_type,
                                        self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm,
                                            use_sigmoid, opt.init_type,opt.gpu_ids)

        if self.isTrain:
            #define loss functions
            def loss_function():
                pass

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





#change the architecture as my approack drawed