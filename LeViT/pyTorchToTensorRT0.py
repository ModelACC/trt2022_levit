import os
import ctypes
import numpy as np
import torch as t
import onnx
import onnx_graphsurgeon as gs
from cuda import cudart
import tensorrt as trt
import time

onnxFile = "./onnx_models/model.onnx"
onnxSurgeonFile = "./onnx_models/model-surgeon.onnx"
soFile = "./plugin/softmaxPlugin.so"
trtFile = "./trt_plans/model.plan"
resultFile = "./profile_result.txt"
nIn,xIn, tIn, cIn = 8,3,196,196
epsilon = 1e-5
np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

class MyProfiler(trt.IProfiler):
    def __init__(self):
        super(MyProfiler, self).__init__()
        self.f = open(resultFile, "w")

    def report_layer_time(self, layerName, ms):
        self.f.write("Timing: %8.3fus -> %s\n"%(ms*1000, layerName))
        print("Timing: %8.3fus -> %s"%(ms*1000, layerName))

# pyTorch 中创建网络并保存为 .pt 文件 ----------------------------------------------
class Net(t.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.Softmax = t.nn.Softmax(dim=-1)

    def forward(self, x):
        # x = t.mul(x, 1.2)
        y = x.softmax(dim=-1)
        # y = self.Softmax(x)
        # y = t.mul(y, 0.8)
        # y = y.softmax(dim=-1)
        # y = t.mul(y, 3)
        return y

net = Net().cuda()
# print("Succeeded building model in pyTorch!")

# 将 .pt 文件转换为 .onnx 文件 ----------------------------------------------------
t.onnx.export(
    net,
    t.randn(nIn,xIn, tIn, cIn, device="cuda"),
    onnxFile,
    # example_outputs=[t.randn(nIn, cIn, 1, 1, device="cuda")],
    input_names=['x'],
    output_names=['y'],
    #do_constant_folding=True,
    verbose=False,
    keep_initializers_as_inputs=True,
    opset_version=12,
    dynamic_axes={"x": {
        0: "nBatchSize"
    }}
)
# print("Succeeded converting model into onnx!")

graph = gs.import_onnx(onnx.load(onnxFile))
graph.inputs[0].shape = ['bs',xIn, tIn, cIn]
graph.outputs[0].shape = ['bs',xIn, tIn, cIn]

# nsoftmax = 0
# for node in graph.nodes:
#     if node.op == 'Softmax':
#         nsoftmax += 1
#         pluginNode = gs.Node("softmax", "Mysoftmax-%d" % nsoftmax, inputs=[node.inputs[0]], outputs=[node.outputs[0]], attrs={"epsilon": 1e-5})
#         graph.nodes.append(pluginNode)
#         node.outputs.clear()

# graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnxSurgeonFile)
# print("Succeeded replacing softmax Plugin node!")


# TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
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
# print("Succeeded finding onnx file!")
with open(onnxSurgeonFile, 'rb') as model:
    if not parser.parse(model.read()):
        print("Failed parsing onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    # print("Succeeded parsing onnx file!")


# for i, layer in enumerate(network):
#     if "MyLayerNorm" in layer.name:
#         layer.precision = trt.DataType.FLOAT
#         print("find layernorm")

inputTensor = network.get_input(0)
profile.set_shape(inputTensor.name, [1,xIn, tIn, cIn], [nIn,xIn, tIn, cIn], [nIn * 2,xIn, tIn, cIn])
config.add_optimization_profile(profile)
engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
# print("Succeeded building engine!")
with open(trtFile, 'wb') as f:
    f.write(engineString)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

context = engine.create_execution_context()
# context.profiler = MyProfiler()
context.set_binding_shape(0, [nIn,xIn, tIn, cIn])
_, stream = cudart.cudaStreamCreate()
# print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
# print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))


seed = np.random.randn(nIn * cIn * tIn)
data = np.random.randn(nIn,xIn,tIn,cIn).astype(np.float32)

inputH0 = np.ascontiguousarray(data.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
_, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
for i in range(10):
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
start_time = time.time()
for i in range(10):
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
end_time = time.time()
cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
cudart.cudaStreamSynchronize(stream)
cudart.cudaStreamDestroy(stream)
cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)

print("Succeeded running model in TensorRT!")
print("Time of original Softmax:", (end_time-start_time)/10)
# net.eval()
# torch_out = net(t.tensor(data)).detach().cpu().numpy()

# print("outputH0:",outputH0)
# print("torch_out:",torch_out)
