import levit 
import levit_c 
import torch 
import tensorrt as trt 
from cuda import cudart
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import trt_build_engine 
import os 
# from datasets import build_dataset

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

class img_dataset(Dataset):
    def build_transform(self):
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        t = []
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)

    def read_img_numpy(self,img_path):
        out = Image.open(img_path)
        out = out.convert("RGB")
        out = out.resize((224,224))
        input_numpy = np.array(out)
        try:
            input_numpy = input_numpy.transpose(0,1,2).squeeze().astype(np.float32)
        except:
            print("error")     
        return input_numpy

    def __init__(self,dir_path):
        self.dir_path = dir_path
        self.image_names = os.listdir(dir_path)
        self.transform = self.build_transform()
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self,idx):
        name = self.image_names[idx]
        ret_numpy_img = self.read_img_numpy(self.dir_path + name)
        ret_numpy_img = self.transform(ret_numpy_img)
        return (idx,ret_numpy_img)

def evaluate(dataset_path, engine_path):
    batch_size = 16
    engine = trt_build_engine.load_engine(engine_path)

    eval_dataset = img_dataset(dataset_path)
    eval_loader = DataLoader(eval_dataset,batch_size=batch_size,shuffle=True)
    data = next(iter(eval_loader))
    idx,imgs = data 
    imgs = imgs.numpy().astype(np.float16)
    imgs = np.random.random(size=(batch_size,3,224,224)).astype(np.float16)
    # print(imgs)


    ret = inference(engine,batch_size,imgs)
    print(ret)
    # for i, data in enumerate(eval_loader):
    #     idx, imgs = data 
    #     imgs = imgs.numpy()
    #     print(np.shape(imgs))
    #     ret = inference(engine,batch_size,imgs.astype(np.float16))
    #     print(ret)

if __name__ =='__main__':
    evaluate("../../data/img/","./trt_plans/model_128S.plan")

