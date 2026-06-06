import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import cv2
from collections import Counter
from pathlib import Path

# ================== CONFIG ==================
MODEL_PATH = "best_float32.tflite"
LABELS_PATH = "Labels.txt"
INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
OUTPUT_IMAGE = "result_with_boxes.jpg"

# ================== Load Labels (120 classes) ==================
def load_labels(path: str):
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            name = line.strip()
            if name:
                labels[i] = name
    return labels

CLASS_NAMES = load_labels(LABELS_PATH)
NUM_CLASSES = len(CLASS_NAMES)
print(f"Loaded {NUM_CLASSES} classes successfully.")

np.random.seed(42)
COLORS = np.random.randint(80, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)

# ================== Load TFLite Model ==================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape : {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# ================== Preprocessing ==================
def preprocess(img_bgr):
    h, w = img_bgr.shape[:2]
    r = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)
    new_h, new_w = int(h * r), int(w * r)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    top = (640 - new_h) // 2
    left = (640 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    input_tensor = canvas.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor, (r, top, left, w, h), img_bgr.copy()

# ================== Detection ==================
def detect_and_visualize(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Cannot load image")
        return

    input_tensor, scale_info, draw_img = preprocess(img)
    r, pad_top, pad_left, orig_w, orig_h = scale_info

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]   # shape (300, 6)

    print(f"Raw output shape: {output.shape}")

    # Output format from Ultralytics TFLite (one-to-one head): 
    # [x1, y1, x2, y2, confidence, class_id]   → already normalized 0-1 and NMS applied
    x1 = output[:, 0]
    y1 = output[:, 1]
    x2 = output[:, 2]
    y2 = output[:, 3]
    confidences = output[:, 4]
    class_ids = output[:, 5].astype(int)

    # Filter by confidence
    mask = confidences >= CONF_THRESHOLD
    x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(x1) == 0:
        print("No objects detected.")
        cv2.imwrite(OUTPUT_IMAGE, draw_img)
        return

    # ====================== IMPORTANT FIX ======================
    # Denormalize from 0-1 (relative to 640x640 padded image) to original image pixels
    # First convert to pixel coords on the 640x640 canvas
    x1 = x1 * 640
    y1 = y1 * 640
    x2 = x2 * 640
    y2 = y2 * 640

    # Then remove padding and apply resize ratio
    x1 = (x1 - pad_left) / r
    y1 = (y1 - pad_top) / r
    x2 = (x2 - pad_left) / r
    y2 = (y2 - pad_top) / r

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    boxes_xyxy = np.clip(boxes_xyxy, 0, [orig_w, orig_h, orig_w, orig_h])

    # ====================== Drawing ======================
    for (bx1, by1, bx2, by2), score, cls_id in zip(boxes_xyxy, confidences, class_ids):
        color = tuple(int(c) for c in COLORS[cls_id % NUM_CLASSES])
        class_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
        label = f"{class_name} {score:.2f}"

        cv2.rectangle(draw_img, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
        
        # Label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)[0]
        cv2.rectangle(draw_img, 
                      (int(bx1), int(by1) - 28), 
                      (int(bx1) + label_size[0] + 10, int(by1)), 
                      color, -1)
        cv2.putText(draw_img, label, (int(bx1) + 5, int(by1) - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

    cv2.imwrite(OUTPUT_IMAGE, draw_img)

    # ====================== Print Results ======================
    count = Counter(CLASS_NAMES.get(int(cid), f"cls_{cid}") for cid in class_ids)

    print("\n" + "═" * 70)
    print("          TFLite DETECTION RESULT")
    print("═" * 70)
    for name, c in sorted(count.items()):
        verb = "is" if c == 1 else "are"
        plural = "s" if c > 1 and not name.endswith("s") else ""   # avoid "personss"
        print(f"There {verb} {c} {name}{plural}")
    print(f"\nTotal detected: {len(class_ids)} objects")
    print(f"Result image saved as: {OUTPUT_IMAGE}")

# ================== MAIN ==================
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test_tflite.py <image.jpg>")
    else:
        detect_and_visualize(sys.argv[1])
        
        

# so now give me full code
# for remember:
# this code:
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
# import numpy as np
# import cv2
# from collections import Counter
# from pathlib import Path
# # ================== CONFIG ==================
# MODEL_PATH = "best_float32.tflite"
# LABELS_PATH = "Full_Labels.txt"
# INPUT_SIZE = (640, 640)
# CONF_THRESHOLD = 0.25
# IOU_THRESHOLD = 0.45
# OUTPUT_IMAGE = "result_with_boxes.jpg"
# # ================== Load Labels (120 classes) ==================
# def load_labels(path: str):
#     labels = {}
#     with open(path, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             name = line.strip()
#             if name:
#                 labels[i] = name
#     return labels
# CLASS_NAMES = load_labels(LABELS_PATH)
# NUM_CLASSES = len(CLASS_NAMES)
# print(f"Loaded {NUM_CLASSES} classes successfully.")
# np.random.seed(42)
# COLORS = np.random.randint(80, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)
# # ================== Load TFLite Model ==================
# interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# print(f"Input shape : {input_details[0]['shape']}")
# print(f"Output shape: {output_details[0]['shape']}")
# # ================== Preprocessing ==================
# def preprocess(img_bgr):
#     h, w = img_bgr.shape[:2]
#     r = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)
#     new_h, new_w = int(h * r), int(w * r)
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#     canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
#     top = (640 - new_h) // 2
#     left = (640 - new_w) // 2
#     canvas[top:top + new_h, left:left + new_w] = resized
#     input_tensor = canvas.astype(np.float32) / 255.0
#     input_tensor = np.expand_dims(input_tensor, axis=0)
#     return input_tensor, (r, top, left, w, h), img_bgr.copy()
# # ================== Detection ==================
# def detect_and_visualize(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         print("Cannot load image")
#         return
#     input_tensor, scale_info, draw_img = preprocess(img)
#     r, pad_top, pad_left, orig_w, orig_h = scale_info
#     # Inference
#     interpreter.set_tensor(input_details[0]['index'], input_tensor)
#     interpreter.invoke()
#     output = interpreter.get_tensor(output_details[0]['index'])[0]   # shape (300, 6)
#     print(f"Raw output shape: {output.shape}")
#     # Output format from Ultralytics TFLite (one-to-one head):
#     # [x1, y1, x2, y2, confidence, class_id]   → already normalized 0-1 and NMS applied
#     x1 = output[:, 0]
#     y1 = output[:, 1]
#     x2 = output[:, 2]
#     y2 = output[:, 3]
#     confidences = output[:, 4]
#     class_ids = output[:, 5].astype(int)
#     # Filter by confidence
#     mask = confidences >= CONF_THRESHOLD
#     x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
#     confidences = confidences[mask]
#     class_ids = class_ids[mask]
#     if len(x1) == 0:
#         print("No objects detected.")
#         cv2.imwrite(OUTPUT_IMAGE, draw_img)
#         return
#     # ====================== IMPORTANT FIX ======================
#     # Denormalize from 0-1 (relative to 640x640 padded image) to original image pixels
#     # First convert to pixel coords on the 640x640 canvas
#     x1 = x1 * 640
#     y1 = y1 * 640
#     x2 = x2 * 640
#     y2 = y2 * 640
#     # Then remove padding and apply resize ratio
#     x1 = (x1 - pad_left) / r
#     y1 = (y1 - pad_top) / r
#     x2 = (x2 - pad_left) / r
#     y2 = (y2 - pad_top) / r
#     boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
#     boxes_xyxy = np.clip(boxes_xyxy, 0, [orig_w, orig_h, orig_w, orig_h])
#     # ====================== Drawing ======================
#     for (bx1, by1, bx2, by2), score, cls_id in zip(boxes_xyxy, confidences, class_ids):
#         color = tuple(int(c) for c in COLORS[cls_id % NUM_CLASSES])
#         class_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
#         label = f"{class_name} {score:.2f}"
#         cv2.rectangle(draw_img, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
#         
#         # Label background
#         label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)[0]
#         cv2.rectangle(draw_img,
#                       (int(bx1), int(by1) - 28),
#                       (int(bx1) + label_size[0] + 10, int(by1)),
#                       color, -1)
#         cv2.putText(draw_img, label, (int(bx1) + 5, int(by1) - 8),
#                     cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
#     cv2.imwrite(OUTPUT_IMAGE, draw_img)
#     # ====================== Print Results ======================
#     count = Counter(CLASS_NAMES.get(int(cid), f"cls_{cid}") for cid in class_ids)
#     print("\n" + "═" * 70)
#     print("          TFLite DETECTION RESULT")
#     print("═" * 70)
#     for name, c in sorted(count.items()):
#         verb = "is" if c == 1 else "are"
#         plural = "s" if c > 1 and not name.endswith("s") else ""   # avoid "personss"
#         print(f"There {verb} {c} {name}{plural}")
#     print(f"\nTotal detected: {len(class_ids)} objects")
#     print(f"Result image saved as: {OUTPUT_IMAGE}")
# # ================== MAIN ==================
# if **name** == "**main**":
#     import sys
#     if len(sys.argv) != 2:
#         print("Usage: python test_tflite.py <image.jpg>")
#     else:
#         detect_and_visualize(sys.argv[1])
# it print:
# (torch_gpu) D:\FCDS\Graduation Project\object detection\Data>python test_tflite.py bus.jpg
# Loaded 120 classes successfully.
# C:\Users\Mostafa\anaconda3\envs\torch_gpu\lib\site-packages\tensorflow\lite\python\interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
#     TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
#     See the migration guide
#     for details.
#   warnings.warn(_INTERPRETER_DELETION_WARNING)
# INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
# Input shape : [  1 640 640   3]
# Output shape: [  1 300   6]
# Raw output shape: (300, 6)
# ══════════════════════════════════════════════════════════════════════
#           TFLite DETECTION RESULT
# ══════════════════════════════════════════════════════════════════════
# There are 4 0 persons
# There is 1 5 bus
# Total detected: 5 objects
# Result image saved as: result_with_boxes.jpg
# and the result_with_boxes.jpg is right 'it display the boxes very well'
# so now there is just one issue 'class indes'
# Note: Full_Labels.txt:
# 0 person
# 1 bicycle
# 2 car
# ...
# 118 cart
# 119 pot