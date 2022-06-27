import os
import numpy as np
from cuda import cudart
import tensorrt as trt

class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibrationDataFilename, calibrationCount, inputShape, cacheFile, batchsize=64):
        trt.IInt8EntropyCalibrator2.__init__(self)
        inputShape = [batchsize] + inputShape
        self.images = np.load(calibrationDataFilename)
        assert batchsize * calibrationCount <= len(self.images)
        self.calibrationCount = calibrationCount
        self.shape = inputShape
        self.index = list(range(self.images.shape[0]))
        self.buffeSize = trt.volume(inputShape) * trt.float32.itemsize
        self.cacheFile = cacheFile
        _, self.dIn = cudart.cudaMalloc(self.buffeSize)
        self.oneBatch = self.batchGenerator(batchsize)
        self.batchsize = batchsize
        
    def __del__(self):
        cudart.cudaFree(self.dIn)

    def batchGenerator(self, batchsize):
        for i in range(self.calibrationCount):
            print("> calibration %d" % i)
            i = self.index[i*batchsize:(i+1)*batchsize]
            yield np.ascontiguousarray(self.images[i])

    def get_batch_size(self):  # do NOT change name
        return self.batchsize

    def get_batch(self, nameList=None, inputNodeName=None):  # do NOT change name
        try:
            images = next(self.oneBatch)
            cudart.cudaMemcpy(
                self.dIn,
                images.ctypes.data,
                self.buffeSize,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            )
            return [self.dIn]
        except StopIteration:
            return None

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")


if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    m = MyCalibrator("./data.npy", 4096//64, [3, 224, 224], "./int8.cache", batchsize=64)
    for i in range(100):
        ret = m.get_batch()
        print(ret)
