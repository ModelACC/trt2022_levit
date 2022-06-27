#!/bin/sh
python pyTorchToTensorRT0.py
python pyTorchToTensorRT.py
python trt_main_softmax.py --model="128S"