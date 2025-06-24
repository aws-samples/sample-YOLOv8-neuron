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

# for multiprocessing compatibility
def process_inference_task(args):
    process_id, batch_size, input_size, model_path, test_data_index = args
    
    # modify core allocation
    core_id = process_id % 64  #allocate cores from 0 to 63
    os.environ['NEURON_RT_VISIBLE_CORES'] = f'{core_id}'
    os.environ['NEURON_RT_CORE_SHARING'] = '1'  # activate core sharing

    
    # tune sleep time based on process ID
    if process_id >= 16:
        time.sleep(0.1)
    
    # load the model
    model = torch.jit.load(model_path)
    
    # creqate input tensor
    input_tensor = torch.zeros(batch_size, 3, input_size, input_size, dtype=torch.float32)
    input_tensor = torch.randint(0, 255, input_tensor.shape, dtype=torch.float32) / 255.0
    
    # perform inference
    latencies = []
    num_inferences = 500  # inference per process
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
        self.summary_results = {}  # to store summary results
        self.batch_model_paths = {}  # store compiled model paths
        
        # load the model based on the name
        if model_name == 'yolo_v8_n':
            self.model = nn.yolo_v8_n()
        elif model_name == 'yolo_v8_m':
            self.model = nn.yolo_v8_m()
        
        # load the pre-trained weights
        weights_path = f'./weights/best-{model_name[-1]}.pt'
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model = checkpoint['model']
            else:
                self.model = checkpoint
                
            # ensure the model is in evaluation mode and float type
            self.model = self.model.float()
            self.model.eval()
            print(f"Successfully loaded model from {weights_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.compiler_args = [
            '--neuroncore-pipeline-cores', '1',  # allocate 1 core per process
            '--fast-math', 'all',
            '--enable-fast-loading-neuron-binaries',
            '--enable-fast-context-switch'
        ]
        
        # generate a list of indices for test data
        self.test_data_indices = list(range(100))

    def compile_model_for_batch_size(self, batch_size):
        # check if the model for this batch size is already compiled
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
            # save the compiled model
            torch.jit.save(model_neuron, model_path)
            print(f"Model compiled and saved to {model_path}")
            return model_path
        except Exception as e:
            print(f"Error during model compilation for batch_size={batch_size}: {e}")
            raise

    def test_concurrent(self, batch_size, num_processes):
        # ensure the batch size is valid
        model_path = self.batch_model_paths.get(batch_size)
        if not model_path or not os.path.exists(model_path):
            model_path = self.compile_model_for_batch_size(batch_size)
        
        # prepare arguments for each process
        args_list = [
            (i, batch_size, self.input_size, model_path, self.test_data_indices) 
            for i in range(num_processes)
        ]
        
        # use multiprocessing to run inference concurrently
        start_time = time.time()
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_processes) as pool:
            all_results = pool.map(process_inference_task, args_list)
        
        # combine results from all processes
        all_latencies = [latency for process_latencies in all_results for latency in process_latencies]
        total_inferences = len(all_latencies)
        total_time = time.time() - start_time
        
        # calculate metrics
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

    def run_benchmark(self, batch_sizes=[1, 2], num_processes=64):
        """运行不同批次大小的基准测试，固定进程数量"""
        # precompile models for all batch sizes
        for batch in batch_sizes:
            self.compile_model_for_batch_size(batch)
        
        # ready to store results
        batch_summary = []
        
        # test each batch size
        for batch in batch_sizes:
            print(f"\n===== Benchmarking with batch size={batch} =====")
            
            # take multiple runs to get average results
            results = []
            for i in range(3):
                result = self.test_concurrent(batch, num_processes)
                results.append(result)
                print(f"Run {i+1}: Throughput = {result['throughput']:.2f} images/sec")
                time.sleep(1)  #  cool down between runs
            
            # calculate average results
            avg_result = {
                key: np.mean([r[key] for r in results])
                for key in results[0].keys()
            }
            
            # save results for this batch size
            self.results[f'batch_{batch}'] = avg_result
            
            # display average results
            print(f"Average Throughput: {avg_result['throughput']:.2f} images/sec")
            print(f"Time per image: {avg_result['time_per_image']:.2f} ms")
            print(f"Batch Latency (avg): {avg_result['avg_latency']*1000:.2f} ms")
            print(f"Batch Latency (p95): {avg_result['p95_latency']*1000:.2f} ms")
            
            # add to summary
            batch_summary.append([
                batch,  # 批次大小 
                f"{avg_result['throughput']:.2f}",  # throughput
                f"{avg_result['time_per_image']:.2f}"  # processing time per image (ms)
            ])
        
        # save summary results
        self.summary_results['batch_comparison'] = batch_summary
        
        # dispay summary results
        self.display_summary()
                
    def display_summary(self):
        print("\n" + "="*60)
        print(f"BATCH SIZE PERFORMANCE SUMMARY FOR {self.model_name}")
        print("="*60)
        
        # get summary data
        summary_data = self.summary_results.get('batch_comparison', [])
        if summary_data:
            table_headers = ["Batch Size", "Throughput (img/s)", "Time per image (ms)"]
            print(tabulate(summary_data, headers=table_headers, tablefmt="grid"))
        
        print("="*60)
        
    def save_results(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = 'benchmark_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # save detailed results
        filename = f'{results_dir}/{self.model_name}_benchmark_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\nDetailed results saved to {filename}")
        
        # save summary results
        summary_filename = f'{results_dir}/{self.model_name}_summary_{timestamp}.json'
        with open(summary_filename, 'w') as f:
            # convert summary results to a serializable format
            serializable_summary = {}
            for key, data in self.summary_results.items():
                serializable_summary[key] = [
                    {"batch_size": row[0], "throughput": row[1], "time_per_image": row[2]} 
                    for row in data
                ]
            json.dump(serializable_summary, f, indent=4)
        print(f"Summary results saved to {summary_filename}")

def main():
    # batch sizes to test
    batch_sizes = [1, 2]
    # number of processes to use
    num_processes = 64
    
    # test YOLOv8-n
    print("\nBenchmarking YOLOv8-n...")
    benchmark_n = BenchmarkInference('yolo_v8_n')
    benchmark_n.run_benchmark(batch_sizes=batch_sizes, num_processes=num_processes)
    benchmark_n.save_results()
    
    # test YOLOv8-m
    print("\nBenchmarking YOLOv8-m...")
    benchmark_m = BenchmarkInference('yolo_v8_m')
    benchmark_m.run_benchmark(batch_sizes=batch_sizes, num_processes=num_processes)
    benchmark_m.save_results()

if __name__ == '__main__':
    main()