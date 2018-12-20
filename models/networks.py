import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type = 'normal', init_gain = 0.02, gpu_ids = []):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids)
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain = init_gain)
    return net

def define_G(input_nc, output_nc, ngf, which_model_netG, norm = 'batch', use_dropout = False,
             init_type = 'normal', gpu_ids = []):
    netG = None
    # norm_layer = get_norm_layer(norm_type = norm)
    norm_layer = functools.partial(nn.BatchNorm2d, affine = True) #fix one param
    if which_model_netG == 'SRN':
        netG = SRNGenerator(input_nc, output_nc, ngf, norm_layer = norm_layer, use_dropout = use_dropout, n_blocks = 0)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)
#def Discriminator
def define_D(input_nc, ndf, which_model_netD, n_layers_D = 3, norm = 'batch',
             use_sigmoid = False, init_type = 'normal', gpu_ids = []):
    netD = None
    norm_layer = functools.partial(nn.BatchNorm2d, affine = True)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers = 3, norm_layer = norm_layer,use_sigmoid = use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    return init_net(netD, init_type, gpu_ids)

#define SRN generator
class SRNGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf = 64, norm_layer = nn.BatchNorm2d, use_dropout = False,
                 n_blocks = 6, padding_type = 'reflect'):
        assert(n_blocks >= 0)
        super(SRNGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func = nn.InstanceNorm2d
        else:
            use_bias = norm_layer = nn.InstanceNorm2d

        model = [nn.ReflectinPad2d(2),
                 nn.Conv2d(input_nc, 16, kernel_size=5, padding=0,
                           bias=use_bias),
                 norm_layer(16),
                 nn.ReLU(True)] #16 * 128 * 128
        #n_downsampling = 2 #stride = 2, so downsampling
        model += [nn.Conv2d(16, 32, kernel_size=5,
                            stride=2, padding=2, bias=use_bias),
                  norm_layer(32),
                  nn.ReLU(True)] #32 * 64 * 64

        model += [nn.Conv2d(32, 64, kernel_size=3,
                            stride=2, padding=1, bias=use_bias),
                  norm_layer(64),
                  nn.ReLU(True)] #64 * 32 *32

        model += [nn.Conv2d(64, 128, kernel_size=3,
                            stride=2, padding=1, bias=use_bias),
                  norm_layer(128),
                  nn.ReLU(True)] # 128 * 16 * 16

        model += [nn.Conv2d(128, 256, kernel_size=3,
                            stride=2, padding=1, bias=use_bias),
                  norm_layer(256),
                  nn.ReLU(True)] #256 * 8 * 8

        model += [nn.Conv2d(256, 512, kernel_size=3,
                            stride=2, padding=1, bias=use_bias),
                  norm_layer(512),
                  nn.ReLU(True)] #512 * 4 * 4

        #add fc layers, reshape !!!
        self.fc = nn.Linear(4*4*512, 4*4*512)

        DeConvModel = [nn.ConvTranspose2d(512, 256, kernel_size=3,
                                          stride=2, padding=1, output_padding=1,
                                          bias=use_bias),
                       norm_layer(256),
                       nn.ReLU(True)] #256 * 8 * 8

        DeConvModel += [nn.ConvTranspose2d(256, 128, kernel_size=3,
                                           stride=2,padding=1, output_padding=1,
                                           bias=use_bias),
                        norm_layer(128),
                        nn.ReLU(True)] #128 * 16 * 16

        DeConvModel += [nn.ConvTranspose2d(128, 64, kernel_size=3,
                                           stride=2, padding=1, output_padding=1,
                                           bias=use_bias),
                        norm_layer(64),
                        nn.ReLU(True)] #64 * 32 * 32

        DeConvModel += [nn.ConvTranspose2d(64, 32, kernel_size=3,
                                           stride=2, padding=1, output_padding=1,
                                           bias=use_bias),
                        norm_layer(32),
                        nn.ReLU(True)] #32 * 64 * 64

        DeConvModel += [nn.ConvTranspose2d(32, 16, kernel_size=3,
                                           stride=2, padding=1, output_padding=1,
                                           bias=use_bias),
                        norm_layer(16),
                        nn.ReLU(True)] # 16 * 128 * 128

        DeConvModel += [nn.ReflectionPad2d(3)]
        DeConvModel += [nn.Conv2d(16, output_nc, kernel_size=7, padding=0)]
        DeConvModel += [nn.Tanh()] # 3 * 128 * 128

        self.model = nn.Sequential(*model)
        self.DeConvModel = nn.Sequential(*DeConvModel)

    def forward(self, *input):
        x = self.model(input) # 512 * 4 * 4
        x = x.view(4 * 4 * 512)
        x = self.fc(x)
        x = x.view(1, 512, 4, 4)
        x = self.DeConvModel(x)
        return x

    #Defines the PatchGAN discriminator with the specified arguments
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf = 64, n_layers=3, norm_layer=nn.BatchNorm2d,use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        #input: 3 * 128 * 128
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)] #64 * 64 * 64

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers): #default n_layers = 3, n = 1, 2
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf*nf_mult),
                nn.LeakyReLU(0.2, True)
            ]  # 128 * 32 * 32 --> 256 * 16 * 16

        nf_mult_prev = nf_mult #4
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ] # 512 * 15 * 15

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        assert (use_sigmoid == False)
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, *input):
        return self.model(input)























