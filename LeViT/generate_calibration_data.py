# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datasets import build_dataset
import random
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='../../imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--count', default=4096, type=int)
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')
    return parser


def main(args):
    print(args)
    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    dataset_val, _ = build_dataset(is_train=False, args=args)
    indices = list(range(len(dataset_val)))
    random.shuffle(indices)
    indices = indices[:args.count]
    images = []
    for i in tqdm(indices):
        images.append(dataset_val[i][0].numpy())
    images = np.array(images)
    print(images.shape)
    np.save("data.npy", images)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
