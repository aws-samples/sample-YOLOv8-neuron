import time
import torch
import torch_neuron  
import concurrent.futures
import numpy as np
import psutil
import json
from datetime import datetime
import os
from nets import nn  
import random
from tabulate import tabulate

import multiprocessing as mp
from functools import partial

# 全局函数，供多进程调用
def process_inference_task(args):
    process_id, batch_size, input_size, model_path, test_data_index = args
    
    # 修改核心分配策略 （手动分配）
    core_id = process_id % 64  #分配到x个核心
    # 不再独占核心
    os.environ['NEURON_RT_VISIBLE_CORES'] = f'{core_id}'
    os.environ['NEURON_RT_CORE_SHARING'] = '1'  # 启用核心共享

    
    # 适当延迟加载，避免同一核心上的进程同时加载
    if process_id >= 16:
        time.sleep(0.1)
    
    # 加载模型
    model = torch.jit.load(model_path)
    
    # 创建测试输入
    input_tensor = torch.zeros(batch_size, 3, input_size, input_size, dtype=torch.float32)
    input_tensor = torch.randint(0, 255, input_tensor.shape, dtype=torch.float32) / 255.0
    
    # 执行多次推理以获取更可靠的指标
    latencies = []
    num_inferences = 500  # 每个进程执行的推理次数
    for _ in range(num_inferences):
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_tensor)
        latency = time.time() - start_time
        latencies.append(latency)
    
    return latencies
    
class BenchmarkInference:
    def __init__(self, model_name, input_size=640):
        self.model_name = model_name
        self.input_size = input_size
        self.device = 'cpu'  # neuron
        self.results = {}
        self.summary_results = {}  # 用于存储汇总结果
        self.batch_model_paths = {}  # 存储不同batch_size对应的模型路径
        
        # 加载模型
        if model_name == 'yolo_v8_n':
            self.model = nn.yolo_v8_n()
        elif model_name == 'yolo_v8_m':
            self.model = nn.yolo_v8_m()
        
        # 加载权重
        weights_path = f'./weights/best-{model_name[-1]}.pt'
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model = checkpoint['model']
            else:
                self.model = checkpoint
                
            # 确保模型为float32类型
            self.model = self.model.float()
            self.model.eval()
            print(f"Successfully loaded model from {weights_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.compiler_args = [
            '--neuroncore-pipeline-cores', '1',  # 每个模型只使用1个核心
            '--fast-math', 'all',
            '--enable-fast-loading-neuron-binaries',
            '--enable-fast-context-switch'
        ]
        
        # 生成测试数据索引 
        self.test_data_indices = list(range(100))

    def compile_model_for_batch_size(self, batch_size):
        """为特定batch_size编译模型"""
        # 检查是否已经编译过这个batch_size的模型
        model_path = f'./neuron_model_{self.model_name}_bs{batch_size}.pt'
        self.batch_model_paths[batch_size] = model_path
        
        if os.path.exists(model_path):
            print(f"Model for batch_size={batch_size} already exists, skipping compilation.")
            return model_path
            
        print(f"Compiling model for batch_size={batch_size}...")
        example_input = torch.zeros(batch_size, 3, self.input_size, self.input_size, dtype=torch.float32)
        
        try:
            model_neuron = torch_neuron.trace(
                self.model, 
                example_input,
                compiler_args=self.compiler_args
            )
            # 保存模型供多进程使用
            torch.jit.save(model_neuron, model_path)
            print(f"Model compiled and saved to {model_path}")
            return model_path
        except Exception as e:
            print(f"Error during model compilation for batch_size={batch_size}: {e}")
            raise

    def test_concurrent(self, batch_size, num_processes):
        """使用多进程执行并发推理"""
        # 确保已为当前batch_size编译模型
        model_path = self.batch_model_paths.get(batch_size)
        if not model_path or not os.path.exists(model_path):
            model_path = self.compile_model_for_batch_size(batch_size)
        
        # 准备每个进程的参数
        args_list = [
            (i, batch_size, self.input_size, model_path, self.test_data_indices) 
            for i in range(num_processes)
        ]
        
        # 使用多进程
        start_time = time.time()
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_processes) as pool:
            all_results = pool.map(process_inference_task, args_list)
        
        # 合并所有进程的结果
        all_latencies = [latency for process_latencies in all_results for latency in process_latencies]
        total_inferences = len(all_latencies)
        total_time = time.time() - start_time
        
        # 计算每张图像的处理时间 (ms)
        throughput = total_inferences * batch_size / total_time
        time_per_image = 1000.0 / throughput  # ms per image
        
        return {
            'total_time': total_time,
            'avg_latency': np.mean(all_latencies),
            'p95_latency': np.percentile(all_latencies, 95),
            'throughput': throughput,
            'time_per_image': time_per_image,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }

    def run_benchmark(self, batch_sizes=[1, 4], num_processes=64):
        """运行不同批次大小的基准测试，固定进程数量"""
        # 首先为所有batch size预编译模型
        for batch in batch_sizes:
            self.compile_model_for_batch_size(batch)
        
        # 准备汇总结果的表格数据
        batch_summary = []
        
        # 测试每个批次大小
        for batch in batch_sizes:
            print(f"\n===== Benchmarking with batch size={batch} =====")
            
            # 运行3次取平均
            results = []
            for i in range(3):
                result = self.test_concurrent(batch, num_processes)
                results.append(result)
                print(f"Run {i+1}: Throughput = {result['throughput']:.2f} images/sec")
                time.sleep(1)  # 冷却时间
            
            # 计算平均值
            avg_result = {
                key: np.mean([r[key] for r in results])
                for key in results[0].keys()
            }
            
            # 保存结果
            self.results[f'batch_{batch}'] = avg_result
            
            # 显示当前配置结果
            print(f"Average Throughput: {avg_result['throughput']:.2f} images/sec")
            print(f"Time per image: {avg_result['time_per_image']:.2f} ms")
            print(f"Batch Latency (avg): {avg_result['avg_latency']*1000:.2f} ms")
            print(f"Batch Latency (p95): {avg_result['p95_latency']*1000:.2f} ms")
            
            # 添加到汇总数据
            batch_summary.append([
                batch,  # 批次大小 
                f"{avg_result['throughput']:.2f}",  # 吞吐量
                f"{avg_result['time_per_image']:.2f}"  # 每张图像处理时间
            ])
        
        # 保存汇总数据用于显示
        self.summary_results['batch_comparison'] = batch_summary
        
        # 显示汇总表格
        self.display_summary()
                
    def display_summary(self):
        """显示汇总结果表格"""
        print("\n" + "="*60)
        print(f"BATCH SIZE PERFORMANCE SUMMARY FOR {self.model_name}")
        print("="*60)
        
        # 获取批次大小比较数据
        summary_data = self.summary_results.get('batch_comparison', [])
        if summary_data:
            table_headers = ["Batch Size", "Throughput (img/s)", "Time per image (ms)"]
            print(tabulate(summary_data, headers=table_headers, tablefmt="grid"))
        
        print("="*60)
        
    def save_results(self):
        """保存测试结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = 'benchmark_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存详细结果
        filename = f'{results_dir}/{self.model_name}_benchmark_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\nDetailed results saved to {filename}")
        
        # 保存汇总结果
        summary_filename = f'{results_dir}/{self.model_name}_summary_{timestamp}.json'
        with open(summary_filename, 'w') as f:
            # 将汇总结果转换为可序列化格式
            serializable_summary = {}
            for key, data in self.summary_results.items():
                serializable_summary[key] = [
                    {"batch_size": row[0], "throughput": row[1], "time_per_image": row[2]} 
                    for row in data
                ]
            json.dump(serializable_summary, f, indent=4)
        print(f"Summary results saved to {summary_filename}")

def main():
    # 批处理大小
    batch_sizes = [1, 4]
    # 固定进程数量为64
    num_processes = 64
    
    # 测试 YOLOv8-n
    print("\nBenchmarking YOLOv8-n...")
    benchmark_n = BenchmarkInference('yolo_v8_n')
    benchmark_n.run_benchmark(batch_sizes=batch_sizes, num_processes=num_processes)
    benchmark_n.save_results()
    
    # 测试 YOLOv8-m
    print("\nBenchmarking YOLOv8-m...")
    benchmark_m = BenchmarkInference('yolo_v8_m')
    benchmark_m.run_benchmark(batch_sizes=batch_sizes, num_processes=num_processes)
    benchmark_m.save_results()

if __name__ == '__main__':
    main()