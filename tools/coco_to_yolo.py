import json
from tqdm import tqdm
import os

def convert_coco_to_yolo(json_file, output_path):
    # ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # read COCO JSON file
    try:
        with open(json_file) as f:
            coco = json.load(f)
    except FileNotFoundError:
        print(f"cannot find the json file, please check the path of the json file"): {json_file}")
        return
    except json.JSONDecodeError:
        print(f"JSON file is not valid: {json_file}")
        return
    
    # create category ID to index mapping
    categories = {cat['id']: idx for idx, cat in enumerate(coco['categories'])}
    
    # create image ID to image mapping
    image_dict = {img['id']: img for img in coco['images']}
    
    # label file
    annotations_dict = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_dict:
            annotations_dict[img_id] = []
        annotations_dict[img_id].append(ann)
    
    # process each image
    for img in tqdm(coco['images'], desc="Converting annotations"):
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']
        
        # get image filename without extension
        filename = os.path.splitext(img['file_name'])[0]
        
        # collect labels for this image
        labels = []
        if img_id in annotations_dict:
            for ann in annotations_dict[img_id]:
                try:
                    # transform COCO bounding box to YOLO format
                    box = ann['bbox']
                    x_center = (box[0] + box[2]/2) / img_w
                    y_center = (box[1] + box[3]/2) / img_h
                    width = box[2] / img_w
                    height = box[3] / img_h
                    
                    cat_idx = categories[ann['category_id']]
                    
                    # ensure values are within [0, 1]
                    x_center = min(max(x_center, 0), 1)
                    y_center = min(max(y_center, 0), 1)
                    width = min(max(width, 0), 1)
                    height = min(max(height, 0), 1)
                    
                    labels.append(f"{cat_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                except KeyError as e:
                    print(f"fail when processing: {e}")
                    continue
        
        # write labels to file
        if labels:
            label_file = os.path.join(output_path, f"{filename}.txt")
            try:
                with open(label_file, 'w') as f:
                    f.write('\n'.join(labels))
            except IOError as e:
                print(f"failto write {label_file}: {e}")

if __name__ == "__main__":
    # ensure the directories exist
    annotations_dir = "annotations"  
    labels_dir = "labels"  
    
    # convert COCO annotations to YOLO format
    print("converting...")
    convert_coco_to_yolo(
        os.path.join(annotations_dir, 'instances_train2017.json'),
        os.path.join(labels_dir, 'train2017')
    )
    
    print("transforming test set...")
    convert_coco_to_yolo(
        os.path.join(annotations_dir, 'instances_val2017.json'),
        os.path.join(labels_dir, 'val2017')
    )