#!/bin/sh
mkdir -p onnx_models
mkdir -p trt_plans
cd ./plugin
make clean && make 
cd ..
python trt_main_build.py --model="128S"