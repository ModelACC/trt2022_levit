#!/bin/sh
mkdir -p onnx_models
mkdir -p trt_plans
cd ./plugin
make clean && make 
cd ..
python3 trt_main_build.py --model="128S"
