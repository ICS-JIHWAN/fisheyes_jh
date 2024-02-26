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
    parser.add_argument('--gpu_id', default=None, type=int, help='GPU device number')
    #
    parser.add_argument('--conf_file', default='./config/cyclegan.py', type=str, help='Experiments description file')
    parser.add_argument('--data_path', default='./data/fisheyes.yaml', type=str, help='Path of dataset')
    #
    parser.add_argument('--input_gray', nargs='?', const=True, default=False, type=bool, help='Set the scale of input image, grey scale when set to True')
    parser.add_argument('--img_size', default=640, type=int, help='The input image size')
    parser.add_argument('--batch', default=2, type=int, help='The number of batch size for a training')
    parser.add_argument('--num_workers', default=0, type=int, help='Multi-process data loading')
    parser.add_argument('--epoch', default=10, type=int, help='The number of training interation')
    #
    parser.add_argument('--gan_mode', default='lsgan', type=str, help='The type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--update_discriminator_step', default=20, type=int, help='Discriminator update step size')
    #
    parser.add_argument('--istrain', nargs='?', const=False, default=True)
    #
    parser.add_argument('--save_dir', default='/storage/jhchoi/fish_eyes/runs', type=str, help='Saving root')
    parser.add_argument('--show_tb', default=50, type=int, help='Output images to tensorboard every few steps during the learning')
    return parser
