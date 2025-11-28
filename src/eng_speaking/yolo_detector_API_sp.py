# yolo_detector_API_sp.py
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

# MAIN DETECTION FUNCTION (English + Beautiful Drawing)
def detect_english(img):
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
        return draw_img, "No objects detected in the image."

    # Convert to corner format
    cx, cy, w, h = boxes.T
    x1 = ((cx - w/2)*640 - left)/r
    y1 = ((cy - h/2)*640 - top)/r
    x2 = ((cx + w/2)*640 - left)/r
    y2 = ((cy + h/2)*640 - top)/r
    boxes_xyxy = np.stack([x1,y1,x2,y2], axis=1)
    boxes_xyxy = np.clip(boxes_xyxy, 0, [orig_w, orig_h, orig_w, orig_h])

    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confs.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
    indices = indices.flatten() if len(indices)>0 else []

    final_classes = class_ids[indices]

    # Build beautiful English speech
    count = Counter(CLASS_NAMES.get(i, "object") for i in final_classes)
    parts = []
    for name, c in sorted(count.items()):
        if c == 1:
            parts.append(f"one {name}")
        else:
            parts.append(f"{c} {name}s" if name != "person" else f"{c} persons")

    if len(parts) == 1:
        speech = f"I see {parts[0]}."
    elif len(parts) == 2:
        speech = f"I see {parts[0]} and {parts[1]}."
    else:
        speech = "I see " + ", ".join(parts[:-1]) + f", and {parts[-1]}."

    total = len(indices)
    speech += f" Total {total} objects detected."

    # Draw beautiful boxes with smart text sizing
    for i, cls_id in enumerate(final_classes):
        x1, y1, x2, y2 = map(int, boxes_xyxy[indices[i]])
        color = tuple(map(int, COLORS[cls_id % len(COLORS)]))
        name = CLASS_NAMES.get(cls_id, "object")
        confidence = confs[indices[i]]

        # Smart font scaling based on box size
        box_height = y2 - y1
        class_scale = max(0.4, min(0.8, box_height / 250))
        score_scale = class_scale * 0.8

        class_thickness = max(1, int(class_scale * 2))
        score_thickness = max(1, int(score_scale * 1.8))

        class_text = name
        score_text = f"{confidence:.2f}"

        (cw, ch), _ = cv2.getTextSize(class_text, cv2.FONT_HERSHEY_DUPLEX, class_scale, class_thickness)
        (sw, sh), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, score_scale, score_thickness)

        total_width = cw + sw + 15
        label_y = max(y1 - 10, ch + sh + 15)

        # Small semi-transparent background
        overlay = draw_img.copy()
        cv2.rectangle(overlay, (x1, label_y - ch - sh - 5), (x1 + total_width + 10, label_y + 5), color, -1)
        cv2.addWeighted(overlay, 0.8, draw_img, 0.2, 0, draw_img)

        # Class name (bigger)
        cv2.putText(draw_img, class_text, (x1 + 10, label_y - sh - 5),
                    cv2.FONT_HERSHEY_DUPLEX, class_scale, (255,255,255), class_thickness, cv2.LINE_AA)

        # Confidence (smaller)
        cv2.putText(draw_img, score_text, (x1 + cw + 20, label_y - 5),
                    cv2.FONT_HERSHEY_DUPLEX, score_scale, (200,255,200), score_thickness, cv2.LINE_AA)

        # Box border
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 4)

    return draw_img, speech