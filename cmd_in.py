import argparse

import sys
import os

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='cycleGAN learning setup for fisheyes', add_help=add_help)
    #
    parser.add_argument('--gpu_id', default=None, type=int, help='GPU device number')
    parser.add_argument('--save', nargs='?', const=True, default=False)
    parser.add_argument('--save_dir', default='debugging', type=str)
    parser.add_argument('--resume', nargs='?', const=True, default=False)
    parser.add_argument('--check_gan', default='patch_gan/V_ru', type=str)
    parser.add_argument('--check_segm', default='segmenter/Ngauss_Nsharpening', type=str)
    parser.add_argument('--epoch', default=100, type=int, help='The number of training interation')
    parser.add_argument('--batch', default=2, type=int, help='The number of batch size for a training')
    parser.add_argument('--num_workers', default=0, type=int, help='Multi-process data loading')
    parser.add_argument('--tb_show', default=50, type=int)
    #
    parser.add_argument('--img_size', default=512, type=int, help='Resized input image')
    parser.add_argument('--normalization', default='vit', type=str, help='Image normalization mode : vit, deit, None')
    parser.add_argument('--n_cls', default=12, type=int, help='The number of segmentation class')
    parser.add_argument('--patch_size', default=16, type=int, help='The input patch size of ViT')
    #
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--betas', default=(0.5, 0.845), type=tuple)
    #`
    return parser
