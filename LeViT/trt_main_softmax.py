import os
import ctypes
import numpy as np
import torch as t
import onnx
import onnx_graphsurgeon as gs
from cuda import cudart
import tensorrt as trt
import time
import levit 
from trt_plugin import speed_test
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(
        'LeViT softmax', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='128S', type=str,
                        help='')
    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    trtFile = "./trt_plans/" +"surgeon_" + args.model + ".plan"
    trtFile_nosurgeon = "./trt_plans/" + args.model + ".plan"
    soFile = "./plugin/softmaxPlugin.so"
    
    
    print("LeViT speed without plugin")
    speed_test(trtFile = trtFile_nosurgeon,soFile = soFile,batch_size = 8, iterations = 10)
    print("LeViT speed with plugin")
    speed_test(trtFile = trtFile,soFile = soFile,batch_size = 8, iterations = 10)

    