import argparse

import sys
import os

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='cycleGAN learning setup for fisheyes', add_help=add_help)
    #
    parser.add_argument('--procname', default='Debugging: fisheyes', type=str)
    parser.add_argument('--gpu_id', default=1, type=int, help='GPU device number')
    #
    parser.add_argument('--conf_file', default='cfg_fisheyes.py', type=str, help='Experiments description file')
    parser.add_argument('--data_path', default='./data/fisheyes.yaml', type=str, help='Path of dataset')
    #
    parser.add_argument('--check-path', default=False, type=bool, help='Check matching of image paths and label paths')
    parser.add_argument('--n_cls', default=12, type=int, help='The number of class at ground truth')
    parser.add_argument('--input_gray', nargs='?', const=True, default=False, type=bool, help='Set the scale of input image, gray scale when set to True')
    parser.add_argument('--img-size', default=512, type=int, help='resized input image')
    parser.add_argument('--crop-size', default=480, type=int, help='???')
    parser.add_argument('--window-size', default=768, type=int, help='window split size for inference')
    parser.add_argument('--window-stride', default=512, type=int, help='window split stride for inference')
    parser.add_argument('--max-ratio', default=2, type=int, help='???')
    parser.add_argument('--normalization', default='vit', type=str, help='Image normalization mode : vit, deit, None')
    parser.add_argument('--patch-size', default=16, type=int, help='The input patch size of ViT')
    parser.add_argument('--dist_rank', default=0, type=int, help='???')
    #
    parser.add_argument('--encoder-name', default='vit_large_patch16_384', type=str, help='The backbone name of encoder')
    #
    parser.add_argument('--amp', default=True, type=bool, help='You want auto casting by training')
    parser.add_argument('--batch', default=1, type=int, help='The number of batch size for a training')
    parser.add_argument('--num_workers', default=0, type=int, help='Multi-process data loading')
    parser.add_argument('--epoch', default=10, type=int, help='The number of training interation')
    #
    parser.add_argument('--gan_mode', default='lsgan', type=str, help='The type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--update_discriminator_step', default=20, type=int, help='Discriminator update step size')
    #
    parser.add_argument('--save', nargs='?', const=True, default=False, help='If you want save model check-point, ')
    parser.add_argument('--istrain', nargs='?', const=True, default=False)
    parser.add_argument('--resume', default='/storage/jhchoi/fish_eyes/segmenter/train_100_SGD/checkpoint.pth', type=str)
    # parser.add_argument('--resume', nargs='?', const=False, default=False)
    #
    parser.add_argument('--save_dir', default='/storage/jhchoi/fish_eyes/merge_model', type=str, help='Saving root')
    parser.add_argument('--show_tb', default=30, type=int, help='Output images to tensorboard every few steps during the learning')
    return parser
