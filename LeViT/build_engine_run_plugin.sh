#!/bin/sh
mkdir -p onnx_models
mkdir -p trt_plans
cd ./plugin
make clean && make 
cd ..
python3 trt_main_build.py --model="128S"

python3 pyTorchToTensorRT0.py
python3 pyTorchToTensorRT.py
python3 trt_main_softmax.py --model="128S"
