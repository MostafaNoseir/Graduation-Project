# yolo_detector.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Must be first — kills the annoying INFO message

import tensorflow as tf
import numpy as np
import cv2
from collections import Counter
from pathlib import Path

# ================== CONFIG ==================
MODEL_PATH = "yolo11n_float32.tflite"
LABELS_PATH = "labels.txt"  # 0 person\n1 bicycle\n...
INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
OUTPUT_IMAGE = "result_with_boxes.jpg"

# ================== Load Labels from labels.txt ==================
def load_labels(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"labels.txt not found at {path}")
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ' ' not in line:
                continue
            idx, name = line.split(' ', 1)
            labels[int(idx)] = name
    return labels

# Load class names: {0: 'person', 1: 'bicycle', ...}
CLASS_NAMES = load_labels(LABELS_PATH)
NUM_CLASSES = len(CLASS_NAMES)

# Random beautiful colors (seeded for consistency)
np.random.seed(42)
COLORS = np.random.randint(80, 255, size=(NUM_CLASSES, 3))

# ================== Load TFLite Model ==================
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

# ================== Detection + Drawing ==================
def detect_and_visualize(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image: {image_path}")
        return

    input_tensor, scale_info, draw_img = preprocess(img)
    r, top, left, orig_w, orig_h = scale_info

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0].T  # (8400, 84)

    # Parse predictions
    boxes = output[:, :4]
    scores = output[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # Filter by confidence
    mask = confidences >= CONF_THRESHOLD
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

    if len(boxes) == 0:
        print("No objects detected.")
        return

    # Convert center format → corners
    cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x1 = ((cx - w/2) * 640 - left) / r
    y1 = ((cy - h/2) * 640 - top) / r
    x2 = ((cx + w/2) * 640 - left) / r
    y2 = ((cy + h/2) * 640 - top) / r

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    boxes_xyxy = np.clip(boxes_xyxy, 0, [orig_w, orig_h, orig_w, orig_h])

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
    if len(indices) > 0:
        indices = indices.flatten()
    else:
        indices = []

    final_boxes = boxes_xyxy[indices]
    final_scores = confidences[indices]
    final_classes = class_ids[indices]

    # Count objects
    count = Counter(CLASS_NAMES.get(cid, "unknown") for cid in final_classes)
    lines = [f"There is 1 {name}" if c == 1 else f"There are {c} {name}s" 
             for name, c in sorted(count.items())]
    result_text = "\n".join(lines) + f"\n\nDetected {len(final_boxes)} objects in total."

    # Draw beautiful boxes
    for (x1, y1, x2, y2), score, cls_id in zip(final_boxes, final_scores, final_classes):
        color = tuple(int(c) for c in COLORS[cls_id % NUM_CLASSES])
        label = f"{CLASS_NAMES.get(cls_id, 'unknown')} {score:.2f}"
        
        cv2.rectangle(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
        cv2.rectangle(draw_img, (int(x1), int(y1) - 30), (int(x1) + label_size[0] + 10, int(y1)), color, -1)
        cv2.putText(draw_img, label, (int(x1) + 5, int(y1) - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    # Save & display
    cv2.imwrite(OUTPUT_IMAGE, draw_img)
    cv2.imshow("YOLOv11 - Press any key to close", draw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Final print
    print("\n" + "═" * 70)
    print(" " * 20 + "YOLOv11 OBJECT DETECTION RESULT")
    print("═" * 70)
    print(result_text)
    print("═" * 70)
    print(f"Result image saved as: {OUTPUT_IMAGE}")

# ================== MAIN ==================
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python yolo_detector.py <image.jpg>")
        print("   Or drag & drop image onto run.bat")
    else:
        detect_and_visualize(sys.argv[1])