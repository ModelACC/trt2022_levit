import os
import ctypes
import numpy as np
import torch as t
import onnx
import onnx_graphsurgeon as gs
from cuda import cudart
import tensorrt as trt
import time
import argparse
from trt_convert_onnx import export_onnx_model
from trt_plugin import build_with_plugin, graph_surgeon

def get_args_parser():
    parser = argparse.ArgumentParser(
        'LeViT build', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='128S', type=str,
                        help='')
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    onnx_path = "./onnx_models/" + args.model + ".onnx"
    surgeon_onnx_path = "./onnx_models/" + "surgeon_" +args.model + ".onnx"
    soFile = "./plugin/softmaxPlugin.so"
    trtFile = "./trt_plans/" +"surgeon_" + args.model + ".plan"
    trtFile_nosurgeon = "./trt_plans/" + args.model + ".plan"
    
    
    export_onnx_model(model_name = args.model,onnx_path = onnx_path)
    graph_surgeon(onnxFile = onnx_path, onnxSurgeonFile = surgeon_onnx_path)
    build_with_plugin(
                    onnxSurgeonFile =surgeon_onnx_path ,
                    soFile = soFile,
                    trtFile = trtFile,
                    min_shape = (2,3,224,224),
                    common_shape = (4,3,224,224),
                    max_shape = (16,3,224,224)
                    )
    build_with_plugin(
                    onnxSurgeonFile =onnx_path ,
                    soFile = soFile,
                    trtFile = trtFile_nosurgeon,
                    min_shape = (2,3,224,224),
                    common_shape = (4,3,224,224),
                    max_shape = (16,3,224,224)
                    )
    
    