import os
from collections import defaultdict, Counter
import pandas as pd

# ================= CONFIG =================
DATASET_DIR = r"D:\FCDS\Graduation Project\object detection\Data\Full_112_Dataset"
NUM_CLASSES = 112

# Load class names from yaml manually (or paste list)
CLASS_NAMES = [
    # paste SAME names from your data.yaml here (112 classes)
]

# ==========================================

label_dir_train = os.path.join(DATASET_DIR, "train", "labels")
label_dir_val = os.path.join(DATASET_DIR, "valid", "labels")

all_label_dirs = [label_dir_train, label_dir_val]

instance_counts = Counter()
image_counts = Counter()

total_images = 0
empty_labels = 0
invalid_labels = 0

for label_dir in all_label_dirs:
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue

        total_images += 1
        path = os.path.join(label_dir, file)

        with open(path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            empty_labels += 1
            continue

        used_classes_in_image = set()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                cls_id = int(parts[0])

                if cls_id < 0 or cls_id >= NUM_CLASSES:
                    invalid_labels += 1
                    continue

                instance_counts[cls_id] += 1
                used_classes_in_image.add(cls_id)

            except:
                invalid_labels += 1

        for cls in used_classes_in_image:
            image_counts[cls] += 1


# ================= RESULTS =================

print("\n========== DATASET SUMMARY ==========")
print(f"Total images: {total_images}")
print(f"Empty label files: {empty_labels}")
print(f"Invalid labels: {invalid_labels}")

print("\n========== CLASS DISTRIBUTION ==========")

data = []
for i in range(NUM_CLASSES):
    name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
    instances = instance_counts[i]
    images = image_counts[i]

    data.append([i, name, instances, images])

df = pd.DataFrame(data, columns=["Class ID", "Class Name", "Instances", "Images"])

# Sort by instances
df_sorted = df.sort_values(by="Instances", ascending=False)

print("\nTop 15 Most Frequent Classes:")
print(df_sorted.head(15))

print("\nBottom 15 Least Frequent Classes (⚠️ important):")
print(df_sorted.tail(15))


# ================= SAVE =================
csv_path = os.path.join(DATASET_DIR, "class_distribution.csv")
df_sorted.to_csv(csv_path, index=False)

print(f"\nCSV saved at: {csv_path}")


# import os
# from collections import Counter

# base = r"D:\FCDS\Graduation Project\object detection\Data\Master_Dataset"
# label_dirs = [
#     os.path.join(base, "train", "labels"),
#     os.path.join(base, "valid", "labels"),
#     # os.path.join(base, "test", "labels")   # optional
# ]

# total_counts = Counter()

# for d in label_dirs:
#     if not os.path.exists(d):
#         continue
#     for txt in os.listdir(d):
#         if not txt.endswith(".txt"):
#             continue
#         with open(os.path.join(d, txt), encoding="utf-8") as f:
#             for line in f:
#                 if line.strip():
#                     cls = int(line.split()[0])
#                     total_counts[cls] += 1

# # Map back to names
# names = [
#     'door', 'stairs', 'elevator', 'escalator', 'pothole', 'pole', 'bump', 'barrier', 'curb',
#     'crosswalk', 'puddle', 'tactile_paving', 'accessibility_symbol', 'ramp', 'wall', 'obstacle',
#     'bus_stop', 'street_light', 'road_damage', 'gutter', 'dustbin', 'footpath', 'white_cane',
#     'crutch', 'warning_column', 'road_turn_left', 'road_turn_right', 'cart', 'tree',
#     'traffic_sign'
# ]

# print("\nClass distribution (all splits):")
# for cls_id, count in sorted(total_counts.items()):
#     name = names[cls_id] if cls_id < len(names) else f"unknown_{cls_id}"
#     print(f"{cls_id:2d} {name:22} : {count:6d} instances")

# print(f"\nTotal instances: {sum(total_counts.values())}")
# print(f"Images with objects: {len(total_counts)} classes have labels")