# YOLOv8模型 用于 AWS Inferentia 

本仓库包含了针对 AWS Inferentia 优化的 YOLOv8 目标检测模型实现，使用 AWS Neuron SDK替代原生的cuda支持。
本项目基于 https://github.com/jahongir7174/YOLOv8-pt 对YOLOv8模型使用PyTorch的实现的前提下，专门对AWS Inf1 family做了适配以便让YOLOv8在缺失cuda的AWS实例上运行。

## 概述

YOLOv8 是一种先进的目标检测模型，提供卓越的性能和准确性。本项目将 YOLOv8 适配到 AWS Inferentia 加速器上高效运行，这是 AWS 专为加速深度学习工作负载而设计的硬件。

本仓库包括：
- YOLOv8 模型实现（n、m 和 x 变体，简单修改参数也可兼容其他变体）
- AWS Neuron SDK 集成，用于 Inferentia 加速
- 训练和推理脚本
- 基准测试工具

## 环境要求

### 硬件
- 基于 AWS Inferentia 的实例（inf1）

### 软件
- Python 3.8
- AWS Neuron SDK
- PyTorch 1.13.1
- torch-neuron

## 环境配置

### 方法一：使用 Conda 环境文件

```bash
# 克隆仓库
git clone https://gitlab.aws.dev/liupinzh/YOLOv8-neuron-benchmark
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

## 预编译模型

本仓库包含YOLOv8-n 和 YOLOv8-m 在batch-size 为1，2，4情况下的预编译模型。这样可以省去运行基准测试是重新编译模型的时间。

## 性能

项目在 `benchmark_results/` 目录中包含基准测试结果。这些结果显示了不同 YOLOv8 变体在 Inferentia 硬件上的性能。
