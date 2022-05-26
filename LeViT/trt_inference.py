import levit 
import levit_c 
import torch 
import tensorrt as trt 
from cuda import cudart
import numpy as np

def inference(engine,batch_size,input_data):
    context = engine.create_execution_context()
    context.set_binding_shape(0,[batch_size, 3, 224,224])

    _, stream = cudart.cudaStreamCreate()
    inputHost = np.ascontiguousarray(input_data.reshape(-1))
    outputHost = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))

    _, inputDevice = cudart.cudaMallocAsync(inputHost.nbytes,stream)
    _, outputDevice = cudart.cudaMallocAsync(outputHost.nbytes,stream)

    
    cudart.cudaMemcpyAsync(inputDevice,inputHost.ctypes.data,inputHost.nbytes,cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,stream)
    
    context.execute_async_v2([int(inputDevice),int(outputDevice)],stream)

    cudart.cudaMemcpyAsync(outputHost.ctypes.data,outputDevice,outputHost.nbytes,cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,stream)
    cudart.cudaStreamSynchronize(stream)

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputDevice)
    cudart.cudaFree(outputDevice)
    return outputHost
