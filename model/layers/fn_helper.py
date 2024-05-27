import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools

from utils.model_utils import init_weights
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


def define_D(input_nc, ndf, netDname, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, device=None):
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
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netDname == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_weights(net, init_type, init_gain, device)


# Vision transformer
def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    num_extra_tokens = 1 + ("dist_token" in state_dict.keys())
    patch_size = model.patch_size
    image_size = model.patch_embed.image_size
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                None,
                (image_size[0] // patch_size, image_size[1] // patch_size),
                num_extra_tokens,
            )
        out_dict[k] = v
    return out_dict
