# yolo_detector_accessible.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# Load labels
def load_labels():
    path = Path(LABELS_PATH)
    if not path.exists():
        raise FileNotFoundError("labels.txt not found!")
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ' ' not in line: continue
            idx, name = line.split(' ', 1)
            labels[int(idx)] = name
    return labels

CLASS_NAMES = load_labels()
COLORS = np.random.randint(80, 255, size=(len(CLASS_NAMES), 3))

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(img):
    h, w = img.shape[:2]
    r = min(INPUT_SIZE[0]/h, INPUT_SIZE[1]/w)
    new_h, new_w = int(h*r), int(w*r)
    resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (new_w, new_h))
    canvas = np.full((640,640,3), 114, dtype=np.uint8)
    top = (640 - new_h)//2
    left = (640 - new_w)//2
    canvas[top:top+new_h, left:left+new_w] = resized
    return np.expand_dims(canvas.astype(np.float32)/255.0, 0), (r, top, left, w, h), img.copy()

def detect_and_get_result(img):
    input_tensor, scale_info, draw_img = preprocess(img)
    r, top, left, orig_w, orig_h = scale_info

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0].T

    boxes = output[:, :4]
    scores = output[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)

    mask = confs >= CONF_THRESHOLD
    boxes, confs, class_ids = boxes[mask], confs[mask], class_ids[mask]

    if len(boxes) == 0:
        text = "No objects detected in the image."
        return draw_img, text

    cx, cy, w, h = boxes.T
    x1 = ((cx - w/2)*640 - left)/r
    y1 = ((cy - h/2)*640 - top)/r
    x2 = ((cx + w/2)*640 - left)/r
    y2 = ((cy + h/2)*640 - top)/r
    boxes_xyxy = np.stack([x1,y1,x2,y2], axis=1).clip(0, [orig_w, orig_h, orig_w, orig_h])

    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confs.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
    indices = indices.flatten() if len(indices)>0 else []

    final_classes = class_ids[indices]
    count = Counter(CLASS_NAMES.get(i, "object") for i in final_classes)

    # BEST PROFESSIONAL SPEECH FOR BLIND USERS
    parts = []
    for name, c in sorted(count.items()):
        if c == 1:
            parts.append(f"one {name}")
        else:
            if name == "person":
                parts.append(f"{c} persons")
            else:
                parts.append(f"{c} {name}s")

    if len(parts) == 1:
        speech = f"I see {parts[0]}."
    elif len(parts) == 2:
        speech = f"I see {parts[0]} and {parts[1]}."
    else:
        speech = "I see " + ", ".join(parts[:-1]) + f", and {parts[-1]}."

    total = len(indices)
    speech += f" In total, {total} objects were detected."

    # Draw boxes
    for i, cls_id in enumerate(final_classes):
        x1,y1,x2,y2 = map(int, boxes_xyxy[indices[i]])
        color = tuple(map(int, COLORS[cls_id % len(COLORS)]))
        name = CLASS_NAMES.get(cls_id, "object")
        cv2.rectangle(draw_img, (x1,y1), (x2,y2), color, 3)
        cv2.putText(draw_img, f"{name} {confs[indices[i]]:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

    return draw_img, speech