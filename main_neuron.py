import argparse
import copy
import csv
import os
import warnings

import pickle

import numpy
import torch

#add neuron import
import torch_neuron

import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


def train(args, params):
    #switch to Neuron if specified
    device = 'cpu' if args.neuron else 'cuda'
    
    # modify model loading - support both n and m models
    if args.model_size == 'm':
        model = nn.yolo_v8_m(len(params['names'].values())).to(device)
    else:
        model = nn.yolo_v8_n(len(params['names'].values())).to(device)
    
    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    p = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)

    optimizer = torch.optim.SGD(p[2], params['lr0'], params['momentum'], nesterov=True)

    optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': p[1]})
    del p

    # Scheduler
    lr = learning_rate(args, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
    with open('../datasets/coco/train2017.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append('../datasets/coco/images/train2017/' + filename)

    dataset = Dataset(filenames, args.input_size, params, True)

    if args.world_size <= 1:
        sampler = None
    else:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    if args.world_size > 1:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # Start training
    best = 0
    num_batch = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)
    with open('weights/step.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'mAP@50', 'mAP'])
            writer.writeheader()
        for epoch in range(args.epochs):
            model.train()

            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            m_loss = util.AverageMeter()
            if args.world_size > 1:
                sampler.set_epoch(epoch)
            p_bar = enumerate(loader)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
            if args.local_rank == 0:
                p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

            optimizer.zero_grad()

            for i, (samples, targets, _) in p_bar:
                x = i + num_batch * epoch  # number of iterations
                samples = samples.cuda().float() / 255
                targets = targets.cuda()

                # Warmup
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, numpy.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [params['warmup_bias_lr'], y['initial_lr'] * lr(epoch)]
                        else:
                            fp = [0.0, y['initial_lr'] * lr(epoch)]
                        y['lr'] = numpy.interp(x, xp, fp)
                        if 'momentum' in y:
                            fp = [params['warmup_momentum'], params['momentum']]
                            y['momentum'] = numpy.interp(x, xp, fp)

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)  # forward
                loss = criterion(outputs, targets)

                m_loss.update(loss.item(), samples.size(0))

                loss *= args.batch_size  # loss scaled by batch_size
                loss *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss).backward()

                # Optimize
                if x % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.4g') % (f'{epoch + 1}/{args.epochs}', memory, m_loss.avg)
                    p_bar.set_description(s)

                del loss
                del outputs

            # Scheduler
            scheduler.step()

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)
                writer.writerow({'mAP': str(f'{last[1]:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3),
                                 'mAP@50': str(f'{last[0]:.3f}')})
                f.flush()

                # Update best mAP
                if last[1] > best:
                    best = last[1]

                # Save model
                ckpt = {'model': copy.deepcopy(ema.ema).half()}

                # Save last, best and delete
                torch.save(ckpt, './weights/last.pt')
                if best == last[1]:
                    best_filename = f'./weights/best-{args.model_size}.pt'
                    torch.save(ckpt, best_filename)
                del ckpt

    if args.local_rank == 0:
        best_filename = f'./weights/best-{args.model_size}.pt'
        util.strip_optimizer(best_filename)  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None):
    #modify device setting
    device = 'cpu' if args.neuron else 'cuda'
    
    filenames = []
    with open('../datasets/coco/val2017.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append('../datasets/coco/images/val2017/' + filename)

    dataset = Dataset(filenames, args.input_size, params, False)

    #modify batch size and num_workers
    loader = data.DataLoader(dataset, 1, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    if model is None:
        # modify model loading - use model size specific weights
        best_filename = f'./weights/best-{args.model_size}.pt'
        model = torch.load(best_filename, map_location=device)['model'].float()

        if args.neuron:
            # convert model to Neuron format
            print('Converting model to Neuron format...')
            example_input = torch.zeros([1, 3, args.input_size, args.input_size])
            compiler_args = [
                '--neuroncore-pipeline-cores', '1',
                '--fast-math', 'all',
                '--enable-fast-loading-neuron-binaries',
                '--enable-fast-context-switch'
            ]
            model_neuron = torch.neuron.trace(model, example_input, compiler_args=compiler_args)
            model = model_neuron
            # save the Neuron model
            model_neuron.save("yolov8_neuron.pt")

    #modify model to device
    if not args.neuron:
        model.half()
    model.eval()

    # Configure
    #iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    iou_v = torch.linspace(0.5, 0.95, 10).to(device)
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))
    for samples, targets, shapes in p_bar:
        #samples = samples.cuda()
        #targets = targets.cuda()
        #samples = samples.half()  # uint8 to fp16/32
        #samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        #_, _, height, width = samples.shape  # batch size, channels, height, width

        #修改设备和数据类型处理
        samples = samples.to(device)
        targets = targets.to(device)
        if not args.neuron:
            samples = samples.half()
        samples = samples / 255
        _, _, height, width = samples.shape

        # Inference
        outputs = model(samples)

        # NMS
        #targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
        #outputs = util.non_max_suppression(outputs, 0.001, 0.65)
        
        #modify targets to device
        targets[:, 2:] *= torch.tensor((width, height, width, height)).to(device)
        outputs = util.non_max_suppression(outputs, 0.001, 0.65)
        
        # Metrics
        for i, output in enumerate(outputs):
            labels = targets[targets[:, 0] == i, 1:]
            #correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).to(device)

            if output.shape[0] == 0:
                if labels.shape[0]:
                    #metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                    metrics.append((correct, *torch.zeros((3, 0)).to(device)))
                continue

            detections = output.clone()
            util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

            # Evaluate
            if labels.shape[0]:
                tbox = labels[:, 1:5].clone()  # target boxes
                tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                correct = numpy.zeros((detections.shape[0], iou_v.shape[0]))
                correct = correct.astype(bool)

                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                correct_class = t_tensor[:, 0:1] == detections[:, 5]
                for j in range(len(iou_v)):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                        matches = matches.cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
            metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)

    # Print results
    print('%10.3g' * 3 % (m_pre, m_rec, mean_ap))

    # Return results
    model.float()  # for training
    return map50, mean_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    # add neuron specific arguments
    parser.add_argument('--neuron', action='store_true', help='use AWS Neuron')
    parser.add_argument('--neuron-threads', type=int, default=4,
                      help='number of neuron threads')
    parser.add_argument('--model-size', choices=['n', 'm'], default='n',
                      help='YOLOv8 model size: n (nano) or m (medium)')

    args = parser.parse_args()

    #args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    #args.world_size = int(os.getenv('WORLD_SIZE', 1))

    #modify environment variables for Neuron
    if args.neuron:
        os.environ['NEURON_RT_NUM_CORES'] = str(args.neuron_threads)
    
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    
    #if args.world_size > 1:
        #torch.cuda.set_device(device=args.local_rank)
        #torch.distributed.init_process_group(backend='nccl', init_method='env://')

    #modify distributed training setup for Neuron
    if args.world_size > 1 and not args.neuron:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        
    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)


if __name__ == "__main__":
    main()
