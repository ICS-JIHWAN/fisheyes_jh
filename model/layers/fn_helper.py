import torch
import torch.nn as nn
import functools

from utils.torch_utils import init_weights
from model.layers.generators import ResnetGenerator
from model.layers.discriminators import NLayerDiscriminator, PixelDiscriminator


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer {} is not found'.format(norm_type))
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01,
                                                               patience=5)
    elif opt.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.sch_niter, eta_min=opt.lr_min)
    else:
        return NotImplementedError('learning rate policy {} is not implemented', opt.optim)
    return scheduler


def define_G(input_nc, output_nc, ngf, netGname, norm='batch', use_dropout=False,
             init_type='normal', init_gain=0.02, device=None):
    """Create a generator
        Parameters:
            input_nc (int) -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            ngf (int) -- the number of filters in the last conv layer
            netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
            norm (str) -- the name of normalization layers used in the network: batch | instance | none
            use_dropout (bool) -- if use dropout layers.
            init_type (str)    -- the name of our initialization method.
            init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
            gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        Returns a generator
        two types of generators:
            U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
            Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
            Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
            We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
        The generator has been initialized by <init_net>. It uses RELU for non-linearity.
        """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netGname == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netGname == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)
    return init_weights(net, init_type, init_gain, device)


def define_D(input_nc, ndf, netDname, n_layer_D=3, norm='batch', init_type='normal', init_gain=0.02, device=None):
    """Create a discriminator
        Parameters:
            input_nc (int)     -- the number of channels in input images
            ndf (int)          -- the number of filters in the first conv layer
            netD (str)         -- the architecture's name: basic | n_layers | pixel
            n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
            norm (str)         -- the type of normalization layers used in the network.
            init_type (str)    -- the name of the initialization method.
            init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
            gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        Returns a discriminator
        Three types of discriminators:
            1. [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
            It can classify whether 70×70 overlapping patches are real or fake.
            Such a patch-level discriminator architecture has fewer parameters
            than a full-image discriminator and can work on arbitrarily-sized images
            in a fully convolutional fashion.
            2. [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
            with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
            3. [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
            It encourages greater color diversity but has no effect on spatial statistics.
        The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
        """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netDname == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netDname == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layer_D, norm_layer=norm_layer)
    elif netDname == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_weights(net, init_type, init_gain, device)
