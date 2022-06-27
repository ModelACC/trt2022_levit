import tensorrt as trt 
import numpy as np
import calibrator
import argparse

def setup_engine(
    max_workspace_size_n=23,
    onnx_path="./onnx_models/model_384.onnx",
    trtfile="./trt_plans/model_384_int8_1_16_64_fastest.plan",
    min_shape=(1,3,224,224),
    common_shape=(16,3,224,224),
    max_shape=(64,3,224,224),
    enable_fp16=False,
    enable_int8=False,
):
    # INT8 calibration
    calibrationDataFilename = "./data.npy"
    cacheFile = "./int8.cache"
    calibrationCount = np.load(calibrationDataFilename).shape[0] // common_shape[0]

    print("ONNX model:", onnx_path)
    print("TensorRT model:", trtfile)
    print("Calibration data:", calibrationDataFilename)

    logger = trt.Logger()
    builder = trt.Builder(logger)
    profile = builder.create_optimization_profile()

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size_n << 30
    if enable_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if enable_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        config.int8_calibrator = calibrator.MyCalibrator(
            calibrationDataFilename,
            calibrationCount,
            [3, 224, 224],
            cacheFile,
            batchsize=common_shape[0],
        )
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network,logger)
    with open(onnx_path, "rb") as model:
        parser.parse(model.read())

    if enable_int8:
        for i, layer in enumerate(network):
            if "Sigmoid" in layer.name:
                print(layer.name)
                layer.precision = trt.DataType.HALF

    input_tensor = network.get_input(0)
    profile.set_shape(
        input_tensor.name,
        min_shape,
        common_shape,
        max_shape
    )
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network,config)
    if(engineString == None):
        print("Fail building")
        return 
    print("Success !") 
    with open(trtfile,'wb') as f:
        f.write(engineString)
    return engineString, logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-path', default='./onnx_models/model_384.onnx', type=str)
    parser.add_argument('--engine-path', default='./trt_plans/engine.plan', type=str)
    parser.add_argument('--enable-fp16', action='store_true')
    parser.add_argument('--enable-int8', action='store_true')
    args = parser.parse_args()
    setup_engine(
        onnx_path=args.onnx_path,
        trtfile=args.engine_path,
        enable_fp16=args.enable_fp16,
        enable_int8=args.enable_int8,
    )
