import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import conv, deconv


class Generator(nn.Module):
    # initializers
    def __init__(self, out_ch=3):
        super(Generator, self).__init__()
        # Unet encoder
        self.conv1 = conv(3, 64, 4, bn=False, activation='lrelu')  # (B, 64, 128, 128)
        self.conv2 = conv(64, 128, 4, activation='lrelu')  # (B, 128, 64, 64)
        self.conv3 = conv(128, 256, 4, activation='lrelu')  # (B, 256, 32, 32)
        self.conv4 = conv(256, 512, 4, activation='lrelu')  # (B, 512, 16, 16)
        self.conv5 = conv(512, 512, 4, activation='lrelu')  # (B, 512, 8, 8)
        self.conv6 = conv(512, 512, 4, activation='lrelu')  # (B, 512, 4, 4)
        self.conv7 = conv(512, 512, 4, activation='lrelu')  # (B, 512, 2, 2)
        self.conv8 = conv(512, 512, 4, bn=False, activation='relu')  # (B, 512, 1, 1)

        # Unet decoder
        self.deconv1 = deconv(512, 512, 4, activation='relu')  # (B, 512, 2, 2)
        self.deconv2 = deconv(1024, 512, 4, activation='relu')  # (B, 512, 4, 4)
        self.deconv3 = deconv(1024, 512, 4, activation='relu')  # (B, 512, 8, 8)
        self.deconv4 = deconv(1024, 512, 4, activation='relu')  # (B, 512, 16, 16)
        self.deconv5 = deconv(1024, 256, 4, activation='relu')  # (B, 256, 32, 32)
        self.deconv6 = deconv(512, 128, 4, activation='relu')  # (B, 128, 64, 64)
        self.deconv7 = deconv(256, 64, 4, activation='relu')  # (B, 64, 128, 128)
        self.deconv8 = deconv(128, out_ch, 4, activation='tanh')  # (B, 3, 256, 256)

    # forward method
    def forward(self, input):
        # Unet encoder
        e1 = self.conv1(input)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)
        e8 = self.conv8(e7)

        # Unet decoder
        d1 = F.dropout(self.deconv1(e8), 0.5, training=True)
        d2 = F.dropout(self.deconv2(torch.cat([d1, e7], 1)), 0.5, training=True)
        d3 = F.dropout(self.deconv3(torch.cat([d2, e6], 1)), 0.5, training=True)
        d4 = self.deconv4(torch.cat([d3, e5], 1))
        d5 = self.deconv5(torch.cat([d4, e4], 1))
        d6 = self.deconv6(torch.cat([d5, e3], 1))
        d7 = self.deconv7(torch.cat([d6, e2], 1))
        output = self.deconv8(torch.cat([d7, e1], 1))

        return output


class Discriminator(nn.Module):
    # initializers
    def __init__(self, n_cls):
        super(Discriminator, self).__init__()
        self.conv1 = conv(3+n_cls, 64, 4, bn=False, activation='lrelu')
        self.conv2 = conv(64, 128, 4, activation='lrelu')
        self.conv3 = conv(128, 256, 4, activation='lrelu')
        self.conv4 = conv(256, 512, 4, 1, 1, activation='lrelu')
        self.conv5 = conv(512, 1, 4, 1, 1, activation='none')

    # forward method
    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        return out
