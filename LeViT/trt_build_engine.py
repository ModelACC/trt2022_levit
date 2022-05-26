import levit 
import levit_c 
import torch 
import tensorrt as trt 
from cuda import cudart
import numpy as np

def setup_engine(max_batch_size = 512,
                 max_workspace_size_n = 4,
                 onnx_path = "./onnx_models/levit_128_onnx.onnx",
                 trtfile = "./trt_plans/my_model.plan",
                 min_shape = (2,3,224,224),
                 common_shape = (4,3,224,224),
                 max_shape = (16,3,224,224)
                 ):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    builder.max_batch_size = max_batch_size

    profile = builder.create_optimization_profile()

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size_n<<30
    config.set_flag(trt.BuilderFlag.FP16)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    parser = trt.OnnxParser(network,logger)
    with open(onnx_path, "rb") as model:
        parser.parse(model.read())

    input_tensor = network.get_input(0)
    profile.set_shape(input_tensor.name,min_shape,common_shape,max_shape)
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network,config)
    if(engineString == None):
        print("Fail building")
        return 
    print("Success !") 
    with open(trtfile,'wb') as f:
        f.write(engineString)
    return engineString, logger

def load_engine(engine_path):
    with open(engine_path,"rb") as f:
        engine_data = f.read()
    engine = trt.Runtime(trt.Logger(trt.Logger.VERBOSE)).deserialize_cuda_engine(engine_data)
    return engine

if __name__ == '__main__':
    setup_engine(onnx_path = "./onnx_models/model_128S.onnx",trtfile = "./trt_plans/model_128S.plan")

