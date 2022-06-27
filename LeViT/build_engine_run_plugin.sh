#!/bin/sh
mkdir -p onnx_models
mkdir -p trt_plans
cd ./plugin
make clean && make 
cd ..
python trt_main_build.py --model="128S"

python pyTorchToTensorRT0.py
python pyTorchToTensorRT.py
python trt_main_softmax.py --model="128S"