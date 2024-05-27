import copy
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import functools

from torchvision import models


class BasicBlock(nn.Module):
    expansion_factor = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu2 = nn.ReLU()
        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion_factor:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion_factor))

    def forward(self, x: Tensor) -> Tensor:
        out = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += self.residual(out)
        x = self.relu2(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self._init_layer()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion_factor
        return nn.Sequential(*layers)

    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class UpsampleNet(nn.Module):
    def __init__(self):
        super(UpsampleNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            # Conv2d layer 정의
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        self.dec1 = CBR2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.unpool1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2 = CBR2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.unpool2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3 = CBR2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.unpool3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4 = CBR2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.unpool4 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec5 = CBR2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True)
        self.unpool5 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec6 = CBR2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

        self._init_layer()

    def forward(self, x):
        output = self.dec1(x)
        output = self.unpool1(output)
        output = self.dec2(output)
        output = self.unpool2(output)
        output = self.dec3(output)
        output = self.unpool3(output)
        output = self.dec4(output)
        output = self.unpool4(output)
        output = self.dec5(output)
        output = self.unpool5(output)
        output = self.dec6(output)

        return output

    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResnetGenerator(nn.Module):
    def __init__(self, num_blocks: list):
        super(ResnetGenerator, self).__init__()
        self.backbone = ResNet(BasicBlock, num_blocks)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, bias=False),
            nn.ReLU()
        )

        self.model_ru = UpsampleNet()

    def forward(self, x):
        output = self.backbone(x)

        output = self.conv1x1(output)

        output = self.model_ru(output)  # B x 1 x H x W

        return output


if __name__ == '__main__':
    model = ResnetGenerator([2, 2, 2, 2])

    input = torch.rand(3, 3, 512, 512)

    model(input)
