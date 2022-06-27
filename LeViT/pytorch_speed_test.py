# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import os
import torch
import torchvision
import time
import timm
import levit
import levit_c
import torchvision
import utils
torch.autograd.set_grad_enabled(False)


T0 = 10
T1 = 60


def compute_throughput_cpu(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    # warmup
    start = time.time()
    while time.time() - start < T0:
        model(inputs)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)


def compute_throughput_cuda(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast():
        while time.time() - start < T0:
            model(inputs)
    timing = []
    if device == 'cuda:0':
        torch.cuda.synchronize()
    with torch.cuda.amp.autocast():
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)


for device in ['cuda:0', 'cpu']:

    if 'cuda' in device and not torch.cuda.is_available():
        print("no cuda")
        continue

    if device == 'cpu':
        os.system('echo -n "nb processors "; '
                  'cat /proc/cpuinfo | grep ^processor | wc -l; '
                  'cat /proc/cpuinfo | grep ^"model name" | tail -1')
        print('Using 1 cpu thread')
        torch.set_num_threads(1)
        compute_throughput = compute_throughput_cpu
    else:
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        compute_throughput = compute_throughput_cuda

    for n, batch_size0, resolution in [
        # ('levit.LeViT_128S', 2048, 224),
        # ('levit.LeViT_128', 2048, 224),
        # ('levit.LeViT_192', 2048, 224),
        # ('levit.LeViT_256', 2048, 224),
        ('levit.LeViT_384', 16, 224),
    ]:
        if device == 'cpu':
            batch_size = 16
        else:
            batch_size = batch_size0
            torch.cuda.empty_cache()
        inputs = torch.randn(batch_size, 3, resolution,
                             resolution, device=device)
        model = eval(n)(num_classes=1000)
        utils.replace_batchnorm(model)
        model.to(device)
        model.eval()

        for i in range(100):
            output = model(inputs)

        start = time.time()
        for i in range(1000):
            output = model(inputs)
        end = time.time()
        print(end-start)
