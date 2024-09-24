#!/usr/bin/env python
# This assumes you're using a Unix-like environment where the python executable is in the environment's path.

import argparse
import os
from trainCGAN import main  # Assuming trainCGAN.py has a function called 'train'

def parse_args():
    parser = argparse.ArgumentParser(description="Train cGAN model.")
    parser.add_argument('--rank', type=int, default=0, help='Rank of the process in distributed training')
    parser.add_argument('--node', type=str, default="0015", help='Node identifier')
    parser.add_argument('--dist-url', type=str, default='tcp://localhost:4321', help='URL for distributed backend')
    parser.add_argument('--dist-backend', type=str, default='nccl', help='Distributed backend')
    parser.add_argument('--world-size', type=int, default=1, help='Number of processes in the distributed environment')
    parser.add_argument('--dataset', type=str, default='mitbith', help='Dataset to use')
    parser.add_argument('--bottom_width', type=int, default=8, help='Bottom width of the generator')
    parser.add_argument('--max_iter', type=int, default=500000, help='Maximum number of iterations')
    parser.add_argument('--img_size', type=int, default=32, help='Image size')
    parser.add_argument('--gen_model', type=str, default='my_gen', help='Generator model name')
    parser.add_argument('--dis_model', type=str, default='my_dis', help='Discriminator model name')
    parser.add_argument('--df_dim', type=int, default=384, help='Feature dimension size in discriminator')
    parser.add_argument('--d_heads', type=int, default=4, help='Number of heads in discriminator')
    parser.add_argument('--d_depth', type=int, default=3, help='Depth of discriminator')
    parser.add_argument('--g_depth', type=str, default='5,4,2', help='Depth of generator')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension size')
    parser.add_argument('--gf_dim', type=int, default=1024, help='Feature dimension size in generator')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of worker threads')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='Learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.0003, help='Learning rate for discriminator')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')
    parser.add_argument('--loss', type=str, default='lsgan', help='Loss function')
    parser.add_argument('--wd', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 hyperparameter for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 hyperparameter for Adam optimizer')
    parser.add_argument('--phi', type=float, default=1.0, help='Phi parameter')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_eval_imgs', type=int, default=50000, help='Number of evaluation images')
    parser.add_argument('--init_type', type=str, default='xavier_uniform', help='Type of weight initialization')
    parser.add_argument('--n_critic', type=int, default=1, help='Number of critic steps per generator step')
    parser.add_argument('--val_freq', type=int, default=20, help='Validation frequency')
    parser.add_argument('--print_freq', type=int, default=50, help='Print frequency')
    parser.add_argument('--grow_steps', type=int, default=None, help='Grow steps for progressive training')
    parser.add_argument('--fade_in', type=float, default=0.0, help='Fade-in duration')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size')
    parser.add_argument('--ema_kimg', type=int, default=500, help='Number of images for EMA update')
    parser.add_argument('--ema_warmup', type=float, default=0.1, help='EMA warmup duration')
    parser.add_argument('--ema', type=float, default=0.9999, help='EMA decay rate')
    parser.add_argument('--diff_aug', type=str, default='translation,cutout,color', help='Differentiable augmentation')
    parser.add_argument('--exp_name', type=str, default='mitbithCGAN', help='Experiment name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--random_seed', type=int, default=1, help="Random seed")
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID')
    parser.add_argument('--distributed', type=bool, default=False, help='Distributed training')
    parser.add_argument('--multiprocessing_distributed', type=bool, default=False, help='Distributed training')
    parser.add_argument('--max_epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--load_path', type=str, default=None, help='Load path')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay')
    return parser.parse_args()


def main_():
    args = parse_args()
    # Pass the entire args namespace or unpack the arguments as needed for the train function
    main(args)

if __name__ == "__main__":
    main_()
