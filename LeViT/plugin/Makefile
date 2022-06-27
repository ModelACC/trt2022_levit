CUDA_PATH   = /usr/local/cuda/
NVCC        = $(CUDA_PATH)/bin/nvcc
TRT_PATH    = /usr/lib/x86_64-linux-gnu/
# SM          = 61
# GENCODE     = -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86
GENCODE     = -gencode=arch=compute_86,code=sm_86
CCFLAG      = -w -std=c++14 -O3 -DNDEBUG -Xcompiler -fPIC $(GENCODE)
SOFLAG      = $(CCFLAG) -shared
INCLUDE     = -I. -I$(CUDA_PATH)/include -I$(TRT_PATH)/include
LDFLAG      = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -L$(TRT_PATH)/lib -lnvinfer

SOURCE_CU   = $(shell find . -name '*.cu')
SOURCE_PY   = './pyTorchToTensorRT.py'
OBJ         = $(shell find . -name *.o)
DEP         = $(OBJ:.o=.d)

-include $(DEP)

all: $(SOURCE_CU:%.cu=%.so)

%.so: %.o
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

%.o: %.cu
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -o $@ -c $<
	
# .PHONY: test
# test:
# 	make clean
# 	python $(SOURCE_PY)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.plan
