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

class BenchmarkInference:
    def __init__(self, model_name, input_size=640):
        self.model_name = model_name
        self.input_size = input_size
        self.device = 'cpu'  # neuron
        self.results = {}
        
        # load the model based on the name
        if model_name == 'yolo_v8_n':
            self.model = nn.yolo_v8_n()
        elif model_name == 'yolo_v8_m':
            self.model = nn.yolo_v8_m()
        
        # load the weights
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
        
        # Convert to Neuron model
        print("Converting to Neuron model...")
        example_input = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
        
        try:
            self.model_neuron = torch_neuron.trace(
                self.model, 
                example_input,
                compiler_args=['--enable-fast-math']  
            )
            print("Model converted successfully")
        except Exception as e:
            print(f"Error during model conversion: {e}")
            print("Attempting conversion without compiler args...")
            try:
                self.model_neuron = torch_neuron.trace(self.model, example_input)
                print("Model converted successfully without compiler args")
            except Exception as e2:
                print(f"Second conversion attempt failed: {e2}")
                raise

    def create_test_input(self, batch_size=1):
        """Create test input"""
        input_tensor = torch.zeros(batch_size, 3, self.input_size, self.input_size, 
                                 dtype=torch.float32)  # Ensure float32 type
        input_tensor = torch.randint(0, 255, input_tensor.shape, 
                                   dtype=torch.float32)  # Generate random data
        input_tensor = input_tensor / 255.0  # Normalize
        return input_tensor

    def single_inference(self, batch_size=1):
        """Single inference"""
        input_tensor = self.create_test_input(batch_size)
        start_time = time.time()
        with torch.no_grad():  # Add no_grad context
            _ = self.model_neuron(input_tensor)
        return time.time() - start_time
    
    def test_concurrent(self, batch_size, num_concurrent):
        """Test concurrent performance"""
        latencies = []
        
        def inference_task():
            return self.single_inference(batch_size)
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(inference_task) for _ in range(num_concurrent)]
            for future in concurrent.futures.as_completed(futures):
                latencies.append(future.result())
                
        total_time = time.time() - start_time
        
        return {
            'total_time': total_time,
            'avg_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'throughput': num_concurrent * batch_size / total_time,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }
    
    def run_benchmark(self, batch_sizes=[1], concurrent_requests=[1,2,4,8,16,32,64]):
        """Run complete benchmark test"""
        for batch in batch_sizes:
            for concurrent in concurrent_requests:
                print(f"\nTesting with batch={batch}, concurrent={concurrent}")
                
                # Run 3 times and take average
                results = []
                for i in range(3):
                    result = self.test_concurrent(batch, concurrent)
                    results.append(result)
                    time.sleep(1)  # Cool down time
                
                # Calculate average values
                avg_result = {
                    key: np.mean([r[key] for r in results])
                    for key in results[0].keys()
                }
                
                # Save results
                self.results[f'batch_{batch}_concurrent_{concurrent}'] = avg_result
                
                print(f"Throughput: {avg_result['throughput']:.2f} images/sec")
                print(f"Avg Latency: {avg_result['avg_latency']*1000:.2f} ms")
                print(f"P95 Latency: {avg_result['p95_latency']*1000:.2f} ms")
                
    def save_results(self):
        """Save test results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = 'benchmark_results'
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f'{results_dir}/{self.model_name}_benchmark_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\nResults saved to {filename}")

def main():
    # Test YOLOv8-n
    print("\nBenchmarking YOLOv8-n...")
    benchmark_n = BenchmarkInference('yolo_v8_n')
    benchmark_n.run_benchmark()
    benchmark_n.save_results()
    
    # Test YOLOv8-m
    print("\nBenchmarking YOLOv8-m...")
    benchmark_m = BenchmarkInference('yolo_v8_m')
    benchmark_m.run_benchmark()
    benchmark_m.save_results()

if __name__ == '__main__':
    main()