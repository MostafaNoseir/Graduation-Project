import os
import shutil
import yaml
import json
from pathlib import Path
from tqdm import tqdm

# ================== CONFIG ==================
base_dir = Path(r'D:\FCDS\Graduation Project\object detection\Data')
combined_dir = base_dir / 'combined_dataset'

# Create folder structure
(combined_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
(combined_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
(combined_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
(combined_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

# ================== 1. COCO 80 classes ==================
coco_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# ================== 2. Your 40 NEW_CLASSES ==================
NEW_CLASSES = [
    'symbol', 'door', 'elevator', 'escalator', 'ramp', 'stairs',
    'curb', 'crosswalk', 'bus_stop',
    'pothole', 'hole', 'bump', 'crack', 'puddle',
    'pole', 'wall', 'obstacle', 'barrier',
    'buildings', 'window', 'dustbin', 'tree',
    'white_cane', 'crutch', 'keys', 'bag', 'desk', 'jacket',
    'pants', 'shoes', 'plate', 'wallet', 'cupboard', 'stick', 'mirror',
    'stove', 'traffic_sign', 'switch_box', 'cart', 'pot'
]

full_class_names = coco_names + NEW_CLASSES
class_id_map = {name: i for i, name in enumerate(full_class_names)}
print(f"✅ Final total classes: {len(full_class_names)} (80 COCO + 40 new)")

# ================== 3. Name normalization map ==================
name_map = {
    'accessibility_symbol': 'symbol', 'doors': 'door', 'closed_door': 'door', 'Door': 'door', 'pintu': 'door',
    'ramps': 'ramp', 'staircase': 'stairs', 'stair': 'stairs', 'tangga': 'stairs',
    'curb': 'curb', 'Curb': 'curb', 'crosswalk': 'crosswalk', 'Crosswalk': 'crosswalk', 'zebracrossing': 'crosswalk',
    'bus_stop': 'bus_stop',
    'pit_hole': 'hole', 'lubang': 'hole', 'Uncovered manhole': 'hole',
    'bump': 'bump', 'crack': 'crack', 'jalan rusak': 'crack', 'puddle': 'puddle', 'Puddle': 'puddle',
    'pole': 'pole', 'Electric pole': 'pole', 'tiang': 'pole',
    'wall': 'wall', 'Wall': 'wall', 'obstacle': 'obstacle', 'hambatan': 'obstacle', 'penghalang jalan': 'obstacle',
    'barrier': 'barrier', 'pagar': 'barrier', 'fence': 'barrier',
    'buildings': 'buildings', 'Buildings': 'buildings', 'window': 'window', 'Window': 'window',
    'dustbin': 'dustbin', 'kotak sampah': 'dustbin', 'waste_container': 'dustbin',
    'tree': 'tree', 'Tree': 'tree', 'pohon': 'tree',
    'whiteCane': 'white_cane', 'crutch': 'crutch',
    'keys': 'keys', 'Keys': 'keys', 'bag': 'bag', 'Bag': 'bag',
    'desk': 'desk', 'Desk': 'desk', 'jacket': 'jacket', 'Jacket': 'jacket',
    'pants': 'pants', 'Pants': 'pants', 'shoes': 'shoes', 'Shoes': 'shoes',
    'plate': 'plate', 'Plate': 'plate', 'wallet': 'wallet', 'Wallet': 'wallet',
    'cupboard': 'cupboard', 'Cupboard': 'cupboard', 'stick': 'stick', 'mirror': 'mirror', 'Mirror': 'mirror',
    'stove': 'stove', 'Traffic signs': 'traffic_sign', 'Switch-Box': 'switch_box',
    'gerobak': 'cart', 'pot': 'pot',
}

# ================== 4. Process COCO (Manual Safe Method) ==================
print("Processing COCO annotations...")

coco_ann_dir = base_dir / 'COCO' / 'annotations_trainval2017'

def coco_to_yolo(json_path, output_label_dir, split_name):
    with open(json_path) as f:
        data = json.load(f)
    
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    for ann in tqdm(data['annotations'], desc=f"Converting {split_name}"):
        image_id = ann['image_id']
        bbox = ann.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        
        img_info = next((img for img in data['images'] if img['id'] == image_id), None)
        if not img_info:
            continue
            
        w, h = img_info['width'], img_info['height']
        x_center = (bbox[0] + bbox[2] / 2) / w
        y_center = (bbox[1] + bbox[3] / 2) / h
        width = bbox[2] / w
        height = bbox[3] / h
        
        coco_name = cat_id_to_name.get(ann['category_id'])
        if coco_name and coco_name in class_id_map:
            new_id = class_id_map[coco_name]
            line = f"{new_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            
            label_file = output_label_dir / f"{img_info['file_name'].replace('.jpg', '.txt')}"
            with open(label_file, 'a') as f:
                f.write(line + '\n')

# Convert train and val
train_label_dir = combined_dir / 'labels' / 'train'
val_label_dir = combined_dir / 'labels' / 'val'

coco_to_yolo(coco_ann_dir / 'instances_train2017.json', train_label_dir, "train2017")
coco_to_yolo(coco_ann_dir / 'instances_val2017.json', val_label_dir, "val2017")

# Copy COCO images (instead of symlink - this avoids permission error)
print("Copying COCO images (this may take some time)...")
coco_train_dir = base_dir / 'COCO' / 'train2017'
coco_val_dir = base_dir / 'COCO' / 'val2017'

for img in tqdm(list(coco_train_dir.glob('*.*')), desc="Copying train images"):
    shutil.copy(img, combined_dir / 'images' / 'train' / img.name)

for img in tqdm(list(coco_val_dir.glob('*.*')), desc="Copying val images"):
    shutil.copy(img, combined_dir / 'images' / 'val' / img.name)

print("COCO processing completed.")

# ================== 5. Process the 11 Roboflow datasets ==================
new_dataset_folders = [
    'accessibility object detection.v1i.yolov11', 'Blind people object detection.v3i.yolov11',
    'Indoor Obstacle Detection.v11i.yolov11', 'Obstacle detection.v11i.yolov11',
    'Obstacles Avoidance Assistance for Visually Impair', 'OOD.v1i.yolov11',
    'Pothole.v1-raw.yolov11', 'risk-detection-1.v2i.yolov11',
    'Visually impaired persons.v6i.yolov11', 'Visually Impaired.v17i.yolov11',
    'visually_impaired.v2i.yolov11'
]

for ds_name in new_dataset_folders:
    ds_path = base_dir / ds_name
    if not ds_path.exists():
        print(f"⚠️ Skipping missing folder: {ds_name}")
        continue

    yaml_path = ds_path / 'data.yaml'
    with open(yaml_path) as f:
        ds_data = yaml.safe_load(f)
    ds_names = ds_data['names']

    for split in ['train', 'valid']:
        img_dir = ds_path / split / 'images'
        lbl_dir = ds_path / split / 'labels'
        if not img_dir.exists():
            continue

        target_split = 'train' if split == 'train' else 'val'
        target_img = combined_dir / 'images' / target_split
        target_lbl = combined_dir / 'labels' / target_split

        print(f"Processing {ds_name} / {split} ...")

        for img_path in img_dir.glob('*.*'):
            txt_path = lbl_dir / (img_path.stem + '.txt')
            if not txt_path.exists():
                continue

            new_lines = []
            with open(txt_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    old_id = int(parts[0])
                    old_name = ds_names[old_id]
                    clean_name = name_map.get(old_name)
                    if clean_name and clean_name in class_id_map:
                        new_id = class_id_map[clean_name]
                        new_lines.append(f"{new_id} {' '.join(parts[1:])}")

            if new_lines:
                shutil.copy(img_path, target_img / img_path.name)
                with open(target_lbl / txt_path.name, 'w') as f:
                    f.write('\n'.join(new_lines) + '\n')

print("\n✅ All Roboflow datasets processed.")

# ================== 6. Create data.yaml ==================
data_yaml = {
    'path': str(combined_dir),
    'train': 'images/train',
    'val': 'images/val',
    'nc': 120,
    'names': full_class_names
}

with open(combined_dir / 'data.yaml', 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print("\n🎉 Merge completed successfully!")
print(f"Total classes: 120")
print(f"Combined dataset location: {combined_dir}")