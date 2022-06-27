# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from timm.models import create_model
from datasets import build_dataset
import trt_inference
import time
from tqdm import tqdm
from timm.utils import accuracy
from collections import defaultdict, deque
import datetime

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


@torch.no_grad()
def evaluate(data_loader, model, device, type):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    if type == 'pytorch':
        # switch to evaluation mode
        model.eval()
    print(f"start evaluation using {type}")
    for images, target in tqdm(data_loader):
        if type == 'pytorch':
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        elif type == 'tensorrt':
            images = images.numpy()
            target = target.numpy()
            output = trt_inference.inference(model, len(images), images)
            output = torch.tensor(output)
            target = torch.tensor(target)
            loss = criterion(output, target)
        else:
            raise NotImplementedError

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print(output.mean().item(), output.std().item())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_args_parser():
    parser = argparse.ArgumentParser(
        'LeViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    # Model parameters
    parser.add_argument('--model', default='LeViT_384', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')
    # Dataset parameters
    parser.add_argument('--data-path', default='../../imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order',
                                 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--type', default="pytorch", type=str)
    parser.add_argument('--engine-path', type=str, required=False)
    return parser


def main(args):
    print(args)
    device = torch.device("cuda:0")

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    dataset_val, _ = build_dataset(is_train=False, args=args)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    print(f"Creating model: {args.model}")
    if args.type == 'pytorch':
        model = create_model(
            args.model,
            num_classes=1000,
            distillation=True,
            pretrained=True,
            fuse=True,
        )
        model.to(device)
        n_parameters = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
    elif args.type == 'tensorrt':
        model = trt_inference.load_engine(args.engine_path)
        print("load tensorrt engine success")
    else:
        raise NotImplementedError
    test_stats = evaluate(data_loader_val, model, device, type=args.type)
    print(
        f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'LeViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
