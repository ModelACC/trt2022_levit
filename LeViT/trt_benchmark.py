import levit
import tensorrt as trt 
import numpy as np
import onnxruntime as rt 
import torch 
import trt_build_engine
from time import time_ns
from cuda import cudart 

def speed_test_pytorch(model_name,batch_size,iterations):
    if(model_name == "128"):
        my_model = levit.LeViT_128(pretrained=True,distillation=True).eval()
    if(model_name == "128S"):
        my_model = levit.LeViT_128S(pretrained=True,distillation=True).eval()
    if(model_name == "192"):
        my_model = levit.LeViT_192(pretrained=True,distillation=True).eval()
    if(model_name == "256"):
        my_model = levit.LeViT_256(pretrained=True,distillation=True).eval()
    if(model_name == "384"):
        my_model = levit.LeViT_384(pretrained=True,distillation=True).eval()
    
    my_model = my_model.cuda()
    
    input_tensor = torch.randn((batch_size,3,224,224)).cuda()
    t0 = time_ns()
    for i in range(iterations):
        _ = my_model(input_tensor)
    t1 = time_ns()
    
    print("PyTorch Inference Time: ",(t1-t0)/1000/1000/iterations)
    
    
    return 
    
    

def speed_test_onnx(onnx_path,batch_size, iterations):
    provider = ['CUDAExecutionProvider']
    session = rt.InferenceSession(onnx_path,providers=provider)
    input_name = session.get_inputs()[0].name 
    output_name = session.get_outputs()[0].name 
    print("Loading ONNX Model Suceess ... ")
    input_numpy = np.random.randn(batch_size,3,224,224).astype(np.float32)
    t0 = time_ns()
    for i in range(iterations):
        _ = session.run([output_name],{input_name:input_numpy})
    t1 = time_ns()
    print("ONNX inference time",(t1-t0)/1000/1000/iterations)
    return 

def speed_test_trt(onnx_path,engine_path,batch_size,iterations):
    engine = trt_build_engine.load_engine(engine_path)
    context = engine.create_execution_context()
    context.set_binding_shape(0,[batch_size,3,224,224])
    
    print("Loading Engine Success ...")
    
    input_numpy = np.random.randn(batch_size,3,224,224).astype(np.float32)
    inputHost = np.ascontiguousarray(input_numpy.reshape(-1))
    outputHost = np.empty(context.get_binding_shape(1),dtype=trt.nptype(engine.get_binding_dtype(1)))
    
    _,inputDevice = cudart.cudaMalloc(inputHost.nbytes)
    _,outputDevice = cudart.cudaMalloc(outputHost.nbytes)
    cudart.cudaMemcpy(inputDevice,inputHost.ctypes.data,inputHost.nbytes,cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    
    print("Warm up ...")
    for i in range(10):
        context.execute_v2([int(inputDevice),int(outputDevice)])
    
    print("Starting Speed Test ...")
    t0 = time_ns()
    for i in range(iterations):
        context.execute_v2([int(inputDevice),int(outputDevice)])
    t1 = time_ns()
    time_for_infer = (t1-t0)/1000/1000/iterations
    
    print("Time for TensorRT inference",time_for_infer)
    cudart.cudaFree(inputDevice)
    cudart.cudaFree(outputDevice)
    return 


if __name__ == "__main__":
    speed_test_pytorch("128S",16,100)
    speed_test_trt(onnx_path="",engine_path="./trt_plans/model_128S.plan",batch_size=16,iterations=100)
    speed_test_onnx(onnx_path="onnx_models/model_128S.onnx",batch_size=16,iterations=100)
