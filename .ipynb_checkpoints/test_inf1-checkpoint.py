import os
import torch
import torch.neuron as neuron
from nets import nn
from utils import util
import yaml
import argparse
import urllib.request

def download_weights():
    """下载预训练权重"""
    weights_dir = 'weights'
    weights_path = os.path.join(weights_dir, 'best.pt')
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        
    if not os.path.exists(weights_path):
        print("下载预训练权重...")
        # 这里需要替换为实际的权重下载链接
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        urllib.request.urlretrieve(url, weights_path)
        print("权重下载完成")
    
    return weights_path

@torch.no_grad()
def test_single_image(image_path):
    try:
        # 加载配置
        with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
            params = yaml.safe_load(f)
        print("配置文件加载成功")
        
        # 加载模型和权重
        model = nn.yolo_v8_n(len(params['names'].values()))
        weights_path = download_weights()
        
        if os.path.exists(weights_path):
            print(f"加载权重文件: {weights_path}")
            ckpt = torch.load(weights_path, map_location='cpu')
            if 'model' in ckpt:
                model.load_state_dict(ckpt['model'].float().state_dict())
            else:
                model.load_state_dict(ckpt.float().state_dict())
        
        model.eval()
        print("模型初始化成功")
        
        # 加载图片
        import cv2
        import numpy as np
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到图片文件: {image_path}")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
            
        print(f"原始图片尺寸: {img.shape}")
        
        # 保存原始图片用于可视化
        original_img = img.copy()
        
        # 预处理
        img = cv2.resize(img, (640, 640))
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)
        img = img / 255.0
        print("图片预处理完成")
        
        # 推理
        outputs = model(img)
        print("推理完成")
        
        # 后处理
        outputs = util.non_max_suppression(outputs, 0.25, 0.45)  # 调整了阈值
        print("后处理完成")
        
        # 可视化结果
        if len(outputs[0]) > 0:
            for det in outputs[0]:
                bbox = det[:4].int().cpu().numpy()
                conf = float(det[4])
                cls_id = int(det[5])
                
                # 画框
                cv2.rectangle(original_img, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            (0, 255, 0), 2)
                
                # 添加标签
                label = f'{params["names"][cls_id]} {conf:.2f}'
                cv2.putText(original_img, label, 
                           (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 255, 0), 2)
            
            cv2.imwrite('result.jpg', original_img)
            print("检测结果已保存到 result.jpg")
        
        return outputs
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return None

if __name__ == "__main__":
    test_image = "test.jpg"
    
    if not os.path.exists(test_image):
        print(f"警告: 测试图片 {test_image} 不存在")
        print("正在下载测试图片...")
        url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
        urllib.request.urlretrieve(url, test_image)
        print("测试图片下载完成")
    
    print("开始测试...")
    results = test_single_image(test_image)
    
    if results is not None and len(results[0]) > 0:
        print("\n检测结果:")
        for i, det in enumerate(results[0]):
            print(f"目标 {i+1}: 位置={det[:4].tolist()}, 置信度={det[4]:.2f}, 类别={int(det[5])}")
    else:
        print("未检测到任何目标")