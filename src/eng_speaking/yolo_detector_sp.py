# yolo_detector_final_speaks.py  ← THIS ONE WILL SPEAK NO MATTER WHAT
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import cv2
from collections import Counter
from pathlib import Path
import sys

# ================== FORCE SPEECH — WORKS ON EVERY WINDOWS PC ==================
import pyttsx3

# Initialize engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1.0)

# Optional: try to use a good voice
voices = engine.getProperty('voices')
for v in voices:
    if "zira" in v.name.lower() or "microsoft" in v.name.lower():
        engine.setProperty('voice', v.id)
        break

def speak_now(text):
    print(f"\nSPEAKING: {text}")
    engine.say(text)
    engine.runAndWait()   # ← BLOCKS UNTIL FINISHED — 100% HEARD!
    print("Speech finished.\n")

# ================== CONFIG ==================
MODEL_PATH = "yolo11n_float32.tflite"
LABELS_PATH = "labels.txt"
INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
OUTPUT_IMAGE = "result_with_boxes.jpg"

# ================== Load Labels ==================
def load_labels(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found!")
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ' ' not in line: continue
            idx, name = line.split(' ', 1)
            labels[int(idx)] = name
    return labels

CLASS_NAMES = load_labels(LABELS_PATH)
np.random.seed(42)
COLORS = np.random.randint(80, 255, size=(len(CLASS_NAMES), 3))

# ================== Load Model ==================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ================== Preprocessing ==================
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

# ================== MAIN ==================
def run(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        speak_now("Error. Image not found.")
        return

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
        result = "No objects detected."
        speak_now(result)
        print(result)
        return

    # Convert boxes
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

    count = Counter(CLASS_NAMES.get(i, "object") for i in final_classes)
    parts = []
    for name, c in sorted(count.items()):
        if c == 1:
            parts.append(f"1 {name}")
        else:
            parts.append(f"{c} {name}s" if name != "person" else f"{c} persons")

    if len(parts) == 1:
        speech = f"There is {parts[0]}."
    elif len(parts) == 2:
        speech = f"There is {parts[0]} and {parts[1]}."
    else:
        speech = "There are " + ", ".join(parts[:-1]) + f", and {parts[-1]}."

    total = len(indices)
    speech += f" Total {total} objects detected."

    # DRAW BOXES
    for i, cls_id in enumerate(final_classes):
        x1,y1,x2,y2 = map(int, boxes_xyxy[indices[i]])
        color = tuple(map(int, COLORS[cls_id % len(COLORS)]))
        name = CLASS_NAMES.get(cls_id, "object")
        cv2.rectangle(draw_img, (x1,y1), (x2,y2), color, 3)
        cv2.putText(draw_img, name, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 3)

    cv2.imwrite(OUTPUT_IMAGE, draw_img)

    # PRINT + SPEAK (blocking = 100% heard)
    print("\n" + "═" * 70)
    print(" YOLOv11n ACCESSIBLE RESULT")
    print("═" * 70)
    print(speech)
    print("═" * 70)
    print(f"Image saved: {OUTPUT_IMAGE}")

    speak_now(speech)  # ← THIS WILL BE HEARD — NO WAY AROUND IT

# ================== RUN ==================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python yolo_detector_final_speaks.py bus.jpg")
    else:
        run(sys.argv[1])