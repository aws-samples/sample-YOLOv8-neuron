import json
from tqdm import tqdm
import os

def convert_coco_to_yolo(json_file, output_path):
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 读取COCO标注
    try:
        with open(json_file) as f:
            coco = json.load(f)
    except FileNotFoundError:
        print(f"找不到文件: {json_file}")
        return
    except json.JSONDecodeError:
        print(f"JSON文件格式错误: {json_file}")
        return
    
    # 创建类别映射
    categories = {cat['id']: idx for idx, cat in enumerate(coco['categories'])}
    
    # 创建图片ID到文件名的映射
    image_dict = {img['id']: img for img in coco['images']}
    
    # 按图片ID组织标注
    annotations_dict = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_dict:
            annotations_dict[img_id] = []
        annotations_dict[img_id].append(ann)
    
    # 处理每张图片
    for img in tqdm(coco['images'], desc="Converting annotations"):
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']
        
        # 获取文件名（不包含扩展名）
        filename = os.path.splitext(img['file_name'])[0]
        
        # 收集该图片的所有标注
        labels = []
        if img_id in annotations_dict:
            for ann in annotations_dict[img_id]:
                try:
                    # 转换边界框格式
                    box = ann['bbox']
                    x_center = (box[0] + box[2]/2) / img_w
                    y_center = (box[1] + box[3]/2) / img_h
                    width = box[2] / img_w
                    height = box[3] / img_h
                    
                    cat_idx = categories[ann['category_id']]
                    
                    # 确保值在0-1范围内
                    x_center = min(max(x_center, 0), 1)
                    y_center = min(max(y_center, 0), 1)
                    width = min(max(width, 0), 1)
                    height = min(max(height, 0), 1)
                    
                    labels.append(f"{cat_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                except KeyError as e:
                    print(f"处理标注时出错: {e}")
                    continue
        
        # 写入标签文件
        if labels:
            label_file = os.path.join(output_path, f"{filename}.txt")
            try:
                with open(label_file, 'w') as f:
                    f.write('\n'.join(labels))
            except IOError as e:
                print(f"写入文件失败 {label_file}: {e}")

if __name__ == "__main__":
    # 设置正确的路径
    annotations_dir = "annotations"  # COCO标注文件所在目录
    labels_dir = "labels"  # 输出标签文件的目录
    
    # 转换训练集和验证集标注
    print("正在转换训练集...")
    convert_coco_to_yolo(
        os.path.join(annotations_dir, 'instances_train2017.json'),
        os.path.join(labels_dir, 'train2017')
    )
    
    print("正在转换验证集...")
    convert_coco_to_yolo(
        os.path.join(annotations_dir, 'instances_val2017.json'),
        os.path.join(labels_dir, 'val2017')
    )