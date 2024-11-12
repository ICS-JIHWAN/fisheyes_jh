import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from cmd_in import get_args_parser
from utils.metrics import gaussuian_filter, create_color_map
from utils.events import save_yaml
from dataset.dataloader import Dataset
from model.patchgan import Generator, Discriminator
from model.segmenter import VisionTransformer, MaskTransformer, Segmenter

STORAGE = '/storage/jhchoi/fish_eyes'


def main(args, device):
    if args.save:
        from torch.utils.tensorboard import SummaryWriter
        save_path = os.path.join(STORAGE, args.save_dir)
        tblogger = SummaryWriter(os.path.join(save_path, 'logging'))
        save_yaml(vars(args), os.path.join(save_path, 'variant.yml'))


if __name__ == '__main__':
    #
    args = get_args_parser().parse_args()
    #
    if args.gpu_id is not None:
        device = torch.device('cuda:{}'.format(args.gpu_id))
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device('cpu')
    #
    main(args, device)
