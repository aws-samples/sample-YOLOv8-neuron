# YOLOv8 Model for AWS Inferentia

This repository contains YOLOv8 object detection model implementation optimized for AWS Inferentia, using AWS Neuron SDK instead of native CUDA support.
This project is based on https://github.com/jahongir7174/YOLOv8-pt PyTorch implementation of YOLOv8 model, specifically adapted for AWS Inf1 family to enable YOLOv8 running on AWS instances without CUDA.

## Overview

YOLOv8 is an advanced object detection model that provides exceptional performance and accuracy. This project adapts YOLOv8 to run efficiently on AWS Inferentia accelerators, hardware designed by AWS specifically for accelerating deep learning workloads.

This repository includes:
- YOLOv8 model implementation (n, m, and x variants, can be compatible with other variants by simple parameter modifications)
- AWS Neuron SDK integration for Inferentia acceleration
- Training and inference scripts
- Benchmarking tools

## Requirement

### Hardware
- AWS Inferentia-based instances (inf1)

### Software
- Python 3.8
- AWS Neuron SDK
- PyTorch 1.13.1
- torch-neuron

## 环境配置

### 方法一：使用 Conda 环境文件

```bash
# 克隆仓库
git clone https://github.com/aws-samples/sample-YOLOv8-neuron
cd YOLOv8-neuron-benchmark

# 创建并激活 Neuron 环境
conda env create -f neuron_env.yml
conda activate neuron_env
```

### 方法二：手动配置

```bash
# 创建 Python 3.8 环境
conda create -n neuron_env python=3.8
source /home/ec2-user/.bashrc
conda activate neuron_env

# 配置 pip 使用 Neuron 仓库
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# 安装必要的包
pip install torch-neuron neuron-cc torch==1.13.1 torchvision
pip install tensorflow==1.15.5.post1 opencv-python pillow
pip install tqdm PyYAML psutil tabulate
```

## 数据集准备

本项目需要 YOLO 格式的 COCO 数据集。您可以自行前往Ultralytics的资源库下载数据集并进行准备。

## 使用方法

### 训练 （weights文件夹有已经训练好的模型，可以跳过这一步）

```bash
# 在 GPU 上训练 YOLOv8-n 模型
python main.py --train

# 使用分布式数据并行训练（多 GPU）
python -m torch.distributed.launch --nproc_per_node=8 main.py --train
```

### 使用 Neuron 进行测试/推理

```bash
# 在 Inferentia 上测试 YOLOv8-n 模型
python main-neuron.py --test --neuron --neuron-threads 4

# 在 Inferentia 上测试 YOLOv8-m 模型
python main-neuron-m.py --test --neuron --neuron-threads 4
```

`--neuron-threads` 参数控制要使用的 Neuron 核心数量。根据您的实例类型进行调整。

### 基准测试

```bash
# 运行基准测试
python benchmark.py #单进程基准测试
python benchmark_mp.py  # 多进程基准测试
```

## 模型变体

本仓库支持多种 YOLOv8 变体：
- YOLOv8-n：Nano 版本（最小，最快）
- YOLOv8-m：Medium 版本（平衡型）
- YOLOv8-x：Extra large 版本（最高精度）

## 性能

项目在 `benchmark_results/` 目录中包含基准测试结果。这些结果显示了不同 YOLOv8 变体在 Inferentia 硬件上的性能。
