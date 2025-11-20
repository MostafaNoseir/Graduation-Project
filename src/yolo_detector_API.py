# yolo_detector_API.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Kills the INFO message

import tensorflow as tf
import numpy as np
import cv2
from collections import Counter
from pathlib import Path

# ================== CONFIG ==================
MODEL_PATH = "yolo11n_float32.tflite"
LABELS_PATH = "labels.txt"
INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# ================== Load Labels ==================
def load_labels(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found!")
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ' ' not in line:
                continue
            idx, name = line.split(' ', 1)
            labels[int(idx)] = name
    return labels

CLASS_NAMES = load_labels(LABELS_PATH)
NUM_CLASSES = len(CLASS_NAMES)
np.random.seed(42)
COLORS = np.random.randint(80, 255, size=(NUM_CLASSES, 3))

# ================== Load Model ==================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
    canvas[top:top+new_h, left:left+new_w] = resized
    
    input_tensor = canvas.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return input_tensor, (r, top, left, w, h), img_bgr.copy()

# ================== CORE DETECTION FUNCTION (for web) ==================
def detect_and_draw_only(image_bgr):
    """Returns: (result_image: np.ndarray, summary_text: str)"""
    input_tensor, scale_info, draw_img = preprocess(image_bgr)
    r, top, left, orig_w, orig_h = scale_info

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0].T  # (8400, 84)

    boxes = output[:, :4]
    scores = output[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences >= CONF_THRESHOLD
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

    if len(boxes) == 0:
        return draw_img, "No objects detected."

    # Convert to corners
    cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x1 = ((cx - w/2) * 640 - left) / r
    y1 = ((cy - h/2) * 640 - top) / r
    x2 = ((cx + w/2) * 640 - left) / r
    y2 = ((cy + h/2) * 640 - top) / r

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    boxes_xyxy = np.clip(boxes_xyxy, 0, [orig_w, orig_h, orig_w, orig_h])

    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
    indices = indices.flatten() if len(indices) > 0 else []

    final_boxes = boxes_xyxy[indices]
    final_scores = confidences[indices]
    final_classes = class_ids[indices]

    # Summary text
    count = Counter(CLASS_NAMES.get(i, "unknown") for i in final_classes)
    lines = [f"There is 1 {name}" if c == 1 else f"There are {c} {name}s" 
             for name, c in sorted(count.items())]
    summary = "\n".join(lines) + f"\n\nDetected {len(final_boxes)} objects in total."

    # Draw boxes
    for (x1, y1, x2, y2), score, cls_id in zip(final_boxes, final_scores, final_classes):
        color = tuple(map(int, COLORS[cls_id % NUM_CLASSES]))
        label = f"{CLASS_NAMES.get(cls_id, 'unknown')} {score:.2f}"
        cv2.rectangle(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
        cv2.rectangle(draw_img, (int(x1), int(y1)-35), (int(x1)+label_size[0]+20, int(y1)), color, -1)
        cv2.putText(draw_img, label, (int(x1)+10, int(y1)-8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)

    return draw_img, summary