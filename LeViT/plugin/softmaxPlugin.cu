#include "softmaxPlugin.h"
#include "softmax.cuh"
using namespace nvinfer1;

PluginFieldCollection    softmaxPluginCreator::fc_ {};
std::vector<PluginField> softmaxPluginCreator::attr_;

// ALIGNPTR
// int8_t *alignPtr(int8_t *ptr, uintptr_t to)
// {
//     uintptr_t addr = (uintptr_t)ptr;
//     if (addr % to)
//     {
//         addr += to - addr % to;
//     }
//     return (int8_t *)addr;
// }

// // NEXTWORKSPACEPTR
// int8_t *nextWorkspacePtr(int8_t *ptr, uintptr_t previousWorkspaceSize)
// {
//     uintptr_t addr = (uintptr_t)ptr;
//     addr += previousWorkspaceSize;
//     return alignPtr((int8_t *)addr, CUDA_MEM_ALIGN);
// }

// template<typename T, int n>
// __global__ void layerNormKernel(T *pInput, T *pOutput, float epsilon)
// {
//     const int tx = threadIdx.x, index = blockIdx.x * n + threadIdx.x;

//     T _x = pInput[index];

//     __shared__ T mean_shared, var_shared;

//     typedef cub::BlockReduce<T, n>               BlockReduce;
//     __shared__ typename BlockReduce::TempStorage temp;
//     T &                                          ref0 = _x;
//     T                                            sum  = BlockReduce(temp).Sum(ref0);
//     // __syncthreads();
//     if (tx == 0)
//         mean_shared = sum / (T)n;
//     __syncthreads();

//     // printf("mean shared: %f\n", mean_shared);

//     T moment = _x - mean_shared;
//     T moment2 = moment * moment;
//     T &ref1 = moment2;
//     T  var  = BlockReduce(temp).Sum(ref1);
//     // __syncthreads();
//     if (tx == 0)
//         var_shared = var / (T)n;
//     __syncthreads();

//     // printf("var shared: %f\n", var_shared);

//     pOutput[index] = moment * (T)rsqrtf(var_shared + (T)epsilon);
// }

int32_t softmaxPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    // #rows
    int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1]*inputDesc[0].dims.d[2];
    // #cols
    int nValuePerBlock = inputDesc[0].dims.d[inputDesc[0].dims.nbDims-1];

    // auto *    mean                  = reinterpret_cast<float *>(workspace);
    // uintptr_t mean_size             = CEIL_TO(nBlock * sizeof(float), CUDA_MEM_ALIGN);
    // auto *    inv_variance          = reinterpret_cast<float *>(nextWorkspacePtr(reinterpret_cast<int8_t *>(mean), mean_size));
    // uintptr_t inv_variance_size     = mean_size;
    if (inputDesc[0].type == DataType::kFLOAT && outputDesc[0].type == DataType::kFLOAT)
    {
        oneflow::cuda::softmax::DirectLoad<float, float> load((float *)inputs[0], nValuePerBlock);
        oneflow::cuda::softmax::DirectStore<float, float> store((float *)outputs[0], nValuePerBlock);
        oneflow::cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), float>(stream, load, store, nBlock, nValuePerBlock);
        // DispatchLogSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols)
    }
    else if (inputDesc[0].type == DataType::kFLOAT && outputDesc[0].type == DataType::kHALF)
    {
        oneflow::cuda::softmax::DirectLoad<float, float> load((float *)inputs[0], nValuePerBlock);
        oneflow::cuda::softmax::DirectStore<float, half> store((half *)outputs[0], nValuePerBlock);
        oneflow::cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), float>(stream, load, store, nBlock, nValuePerBlock);
    }
    else if (inputDesc[0].type == DataType::kHALF && outputDesc[0].type == DataType::kFLOAT)
    {
        oneflow::cuda::softmax::DirectLoad<half, float> load((half *)inputs[0], nValuePerBlock);
        oneflow::cuda::softmax::DirectStore<float, float> store((float *)outputs[0], nValuePerBlock);
        oneflow::cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), float>(stream, load, store, nBlock, nValuePerBlock);
    }
    else if (inputDesc[0].type == DataType::kHALF && outputDesc[0].type == DataType::kHALF)
    {
        oneflow::cuda::softmax::DirectLoad<half, float> load((half *)inputs[0], nValuePerBlock);
        oneflow::cuda::softmax::DirectStore<float, half> store((half *)outputs[0], nValuePerBlock);
        oneflow::cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), float>(stream, load, store, nBlock, nValuePerBlock);
    }
    else {
        printf("[softmaxPlugin ERROR] Should never reach here\n");
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(softmaxPluginCreator);
