import copy
import numpy as np
import torch
import torch.nn as nn
import functools

from torchvision import models


class ResnetGenerator(nn.Module):
    def __init__(
            self,
            num_fci,
            num_lfo
    ):
        super(ResnetGenerator, self).__init__()
        self.backbone = models.resnet34(pretrained=True)

        self.backbone.conv1 = nn.Conv2d(num_fci, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.model_ru = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(96, 96, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(96, 66, kernel_size=4, stride=2, padding=1),
        )

        # self.model_ru = nn.Sequential(
        #     nn.Conv2d(512, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 128, 3, 1, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        #     nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
        #     nn.ConvTranspose2d(16, 2, kernel_size=4, stride=2, padding=1),
        # )

    def forward(self, x):
        output = self.backbone(x)
        output = output.reshape(
            output.shape[0],
            512,
            int(np.sqrt(output.shape[1] / 512)),
            int(np.sqrt(output.shape[1] / 512))
        )
        output = self.model_ru(output)
        # output = torch.nn.functional.tanh(output) * 10

        return output


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
