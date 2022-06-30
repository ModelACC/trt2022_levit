import os
import ctypes
import numpy as np
import torch as t
import onnx
import onnx_graphsurgeon as gs
from cuda import cudart
import tensorrt as trt
import time
import levit 

np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

class MyProfiler(trt.IProfiler):
    def __init__(self):
        super(MyProfiler, self).__init__()
        self.f = open(resultFile, "w")

    def report_layer_time(self, layerName, ms):
        self.f.write("Timing: %8.3fus -> %s\n"%(ms*1000, layerName))
        print("Timing: %8.3fus -> %s"%(ms*1000, layerName))

def build_with_plugin(
                    onnxSurgeonFile,
                    soFile,
                    trtFile,
                    min_shape = (2,3,224,224),
                    common_shape = (4,3,224,224),
                    max_shape = (16,3,224,224)
                    ):
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.max_workspace_size = 3 << 30

    # config.set_flag(trt.BuilderFlag.FP32)
    # config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxSurgeonFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxSurgeonFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing onnx file!")
        
    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, min_shape, common_shape, max_shape)
    config.add_optimization_profile(profile)
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write(engineString)
    return 
    

def load_engine(trtFile,soFile):
    logger = trt.Logger(trt.Logger.ERROR)
    # trt.init_libnvinfer_plugins(logger, '')
    # ctypes.cdll.LoadLibrary(soFile)
    with open(trtFile,"rb") as f:
        engine_data = f.read()
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_data)
    return engine

# def graph_surgeon(onnxFile, onnxSurgeonFile):
#     graph = gs.import_onnx(onnx.load(onnxFile))
#     for node in graph.nodes:
#         if(node.name == "Add_57"):
#             graph.outputs = [node.outputs[0]]


def graph_surgeon(onnxFile, onnxSurgeonFile):
    graph = gs.import_onnx(onnx.load(onnxFile))
    nsoftmax = 0
    for node in graph.nodes:
        if node.op == 'Softmax':
            nsoftmax += 1
            pluginNode = gs.Node("softmax", "Mysoftmax-%d" % nsoftmax, inputs=[node.inputs[0]], outputs=[node.outputs[0]], attrs={"epsilon": 1e-5})
            graph.nodes.append(pluginNode)
            node.outputs.clear()
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), onnxSurgeonFile)
    print("Succeeded replacing softmax Plugin node!")
    return graph

def trt_run(trtFile,soFile,batch_size):
    # logger = trt.Logger(trt.Logger.ERROR)
    # trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)
    engine = load_engine(trtFile,soFile)
    # my_model = levit.LeViT_128S(pretrained=True,distillation=True).eval()
    
    context = engine.create_execution_context()
    context.profiler = MyProfiler()
    context.set_binding_shape(0,[batch_size,3,224,224])
    _, stream = cudart.cudaStreamCreate()
    print("Loading Engine Success ...")
    input_numpy = np.random.randn(batch_size,3,224,224).astype(np.float32)
    inputH0 = np.ascontiguousarray(input_numpy.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)
    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)
    
    my_model = levit.LeViT_128S(pretrained=True,distillation=True).eval()
    torch_out = my_model(t.tensor(input_numpy)).detach().cpu().numpy()
    
    print(outputH0)
    print(torch_out)
    return 

def speed_test(trtFile,soFile,batch_size = 4, iterations = 10):
    ctypes.cdll.LoadLibrary(soFile)
    engine = load_engine(trtFile,soFile)
        
    context = engine.create_execution_context()
    context.set_binding_shape(0,[batch_size,3,224,224])
    _, stream = cudart.cudaStreamCreate()
    print("Loading Engine Success ...")
    input_numpy = np.random.randn(batch_size,3,224,224).astype(np.float32)
    inputH0 = np.ascontiguousarray(input_numpy.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    for _ in range(10):
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaDeviceSynchronize()
    start_t = time.time()
    for i in range(iterations):
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaDeviceSynchronize()
    end_t = time.time()
    print(str((end_t - start_t)/iterations) + "s")
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)
    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)
    
    # my_model = levit.LeViT_128S(pretrained=True,distillation=True).eval()
    # torch_out = my_model(t.tensor(input_numpy)).detach().cpu().numpy()
    
    # print(outputH0)
    # print(torch_out)
    return 
    

if __name__ == "__main__":
    # ptFile = "./model.pt"
    onnxFile = "./onnx_models/model_384.onnx"
    onnxSurgeonFile = "./onnx_models/surgeon_model_384.onnx"
    soFile = "./plugin/softmaxPlugin.so"
    trtFile = "./trt_plans/model_384.plan"
    resultFile = "./profile_result.txt"
    nIn, tIn, cIn = 2, 3, 4
    epsilon = 1e-5
    
    # graph_surgeon(onnxFile, onnxSurgeonFile)
    # build_with_plugin(onnxSurgeonFile = onnxSurgeonFile,
    #                  soFile = soFile,
    #                  trtFile = trtFile,
    #                  min_shape = (2,3,224,224),
    #                  common_shape = (4,3,224,224),
    #                  max_shape = (16,3,224,224)
    #                  )
    # trt_run(trtFile = trtFile,soFile = soFile,batch_size = 2)
    speed_test(trtFile,soFile,batch_size = 16, iterations = 10)
    
    
