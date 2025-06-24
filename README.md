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

## Environment Setup

### Method 1: Using Conda Environment File

```bash
# Clone the repository
git clone https://github.com/aws-samples/sample-YOLOv8-neuron
cd sample-YOLOv8-neuron

# Create and activate Neuron environment
conda env create -f neuron_env.yml
conda activate neuron_env
```

### Method 2: Manual Configuration (Recommended)

```bash
# Create Python 3.8 environment
conda create -n neuron_env python=3.8
source /home/ec2-user/.bashrc
conda activate neuron_env

# Configure pip to use Neuron repository
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install required packages
pip install torch-neuron neuron-cc torch==1.13.1 torchvision
pip install tensorflow==1.15.5.post1 opencv-python pillow
pip install tqdm PyYAML psutil tabulate
```

## Dataset Preparation

This project requires COCO dataset in YOLO format. You can download and prepare the dataset from Ultralytics' repository.

## Usage

### Training (Pre-trained models are available in weights folder, you can skip this step)

```bash
# Train YOLOv8-n model on GPU
python main.py --train

# Use distributed data parallel training (multi-GPU)
python -m torch.distributed.launch --nproc_per_node=8 main.py --train
```

### Testing/Inference with Neuron

```bash
# Test YOLOv8-n model on Inferentia
python main-neuron.py --test --neuron --neuron-threads 4

# Test YOLOv8-m model on Inferentia
python main-neuron.py --test --neuron --neuron-threads 4 --model-size m
```

The `--neuron-threads` parameter controls the number of Neuron cores to use. Adjust according to your instance type.

### Benchmarking

```bash
# Run benchmark tests
python benchmark.py # Single-process benchmark
python benchmark_mp.py  # Multi-process benchmark
```

## Model Variants

This repository supports multiple YOLOv8 variants:
- YOLOv8-n: Nano version (smallest, fastest)
- YOLOv8-m: Medium version (balanced)
- YOLOv8-x: Extra large version (highest accuracy)

## Performance

The project includes benchmark results in the `benchmark_results/` directory. These results show the performance of different YOLOv8 variants on Inferentia hardware.
