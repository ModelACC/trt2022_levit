import levit 
import torch 
import sys
import onnxoptimizer
import onnx
from onnx import shape_inference

def export_onnx_model(model_name):
    onnx_path = f"./onnx_models/model_{model_name}.onnx"
    input_tensor = torch.randn(1,3,224,224)
    if(model_name == "128"):
        my_model = levit.LeViT_128(pretrained=True,distillation=True, fuse=True).eval()
    elif(model_name == "128S"):
        my_model = levit.LeViT_128S(pretrained=True, distillation=True, fuse=True).eval()
    elif(model_name == "192"):
        my_model = levit.LeViT_192(pretrained=True, distillation=True, fuse=True).eval()
    elif(model_name == "256"):
        my_model = levit.LeViT_256(pretrained=True, distillation=True, fuse=True).eval()
    elif(model_name == "384"):
        my_model = levit.LeViT_384(pretrained=True, distillation=True, fuse=True).eval()
    else:
        raise NotImplementedError

    torch.onnx.export(my_model,               # model being run
                  input_tensor,                         # model input (or a tuple for multiple inputs)
                  onnx_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True, 
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})  # whether to execute constant folding for optimization)
    model = onnx.load(onnx_path)
    model = onnxoptimizer.optimize(model)
    print("model optimized")
    onnx.save(shape_inference.infer_shapes(model), onnx_path)
    print("shape inference done")

if __name__ == '__main__':
    export_onnx_model(model_name=sys.argv[-1])
