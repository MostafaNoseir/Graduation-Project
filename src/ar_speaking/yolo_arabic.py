# yolo_arabic.py  ← الحل الأخير والمثالي
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import cv2
from collections import Counter
from pathlib import Path

# ================== إعدادات ==================
MODEL_PATH = "yolo11n_float32.tflite"
LABELS_PATH = "labels.txt"        # نستخدم labels.txt الإنجليزي الأصلي للكتابة على الصورة
LABELS_ARABIC_PATH = "labels_arabic.txt"  # للنطق والنص العربي فقط

INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# تحميل الأسماء الإنجليزية (للكتابة على الصورة)
def load_english_labels():
    path = Path(LABELS_PATH)
    if not path.exists():
        raise FileNotFoundError("labels.txt غير موجود!")
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ' ' not in line: continue
            idx, name = line.split(' ', 1)
            labels[int(idx)] = name.strip()
    return labels

# تحميل الأسماء العربية (للصوت والنص فقط)
def load_arabic_labels():
    path = Path(LABELS_ARABIC_PATH)
    if not path.exists():
        raise FileNotFoundError("labels_arabic.txt غير موجود!")
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ' ' not in line: continue
            idx, name = line.split(' ', 1)
            labels[int(idx)] = name.strip()
    return labels

CLASS_NAMES_EN = load_english_labels()      # للكتابة على الصورة
CLASS_NAMES_AR = load_arabic_labels()       # للنطق والنص العربي

np.random.seed(42)
COLORS = np.random.randint(80, 255, size=(len(CLASS_NAMES_EN), 3))

# تحميل النموذج
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

def detect_arabic(img):
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
        arabic_text = "لم يتم اكتشاف أي شيء في الصورة."
        return draw_img, arabic_text

    # تحويل الصناديق
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

    # === النص العربي الجميل (نفس السابق) ===
    count = Counter(CLASS_NAMES_AR.get(i, "شيء") for i in final_classes)
    parts = []
    for name, c in sorted(count.items()):
        if c == 1:
            parts.append(f"واحد {name}")
        elif c == 2:
            parts.append(f"اثنان {name}")
        elif c <= 10:
            nums = ["", "", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"]
            suffix = "ات" if name.endswith(("ة","ه","ء")) else ""
            parts.append(f"{nums[c]} {name}{suffix}")
        else:
            parts.append(f"{c} {name}")

    if len(parts) == 1:
        arabic_text = f"أرى {parts[0]} فقط."
    elif len(parts) == 2:
        arabic_text = f"أرى {parts[0]} و{parts[1]}."
    else:
        arabic_text = "أرى " + "، ".join(parts[:-1]) + f"، و{parts[-1]}."

    total = len(indices)
    arabic_text += f" تم اكتشاف {total} شيء إجمالاً."

    # === رسم الصناديق + حجم خط مثالي لا يتداخل أبدًا ===
    for i, cls_id in enumerate(final_classes):
        x1, y1, x2, y2 = map(int, boxes_xyxy[indices[i]])
        color = tuple(map(int, COLORS[cls_id % len(COLORS)]))
        en_name = CLASS_NAMES_EN.get(cls_id, "object")
        confidence = confs[indices[i]]

        # حساب حجم الخط تلقائيًا حسب حجم الصندوق
        box_height = y2 - y1
        class_scale = max(0.4, min(0.8, box_height / 250))  # أصغر قليلاً للكلاس
        score_scale = class_scale * 0.8  # أصغر للثقة

        class_thickness = max(1, int(class_scale * 2))
        score_thickness = max(1, int(score_scale * 1.8))

        class_text = en_name
        score_text = f"{confidence:.2f}"

        (cw, ch), _ = cv2.getTextSize(class_text, cv2.FONT_HERSHEY_DUPLEX, class_scale, class_thickness)
        (sw, sh), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, score_scale, score_thickness)

        total_width = cw + sw + 15  # مسافة صغيرة بينهما
        label_y = max(y1 - 10, ch + sh + 15)  # مسافة آمنة

        # خلفية أصغر
        overlay = draw_img.copy()
        cv2.rectangle(overlay, (x1, label_y - ch - sh - 5), (x1 + total_width + 10, label_y + 5), color, -1)  # خلفية أصغر
        cv2.addWeighted(overlay, 0.8, draw_img, 0.2, 0, draw_img)

        # اسم الكلاس (أكبر)
        cv2.putText(draw_img, class_text, (x1 + 10, label_y - sh - 5),
                    cv2.FONT_HERSHEY_DUPLEX, class_scale, (255,255,255), class_thickness, cv2.LINE_AA)

        # الثقة (أصغر، بجانبه)
        cv2.putText(draw_img, score_text, (x1 + cw + 20, label_y - 5),
                    cv2.FONT_HERSHEY_DUPLEX, score_scale, (255,255,255), score_thickness, cv2.LINE_AA)

        # إطار الصندوق
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 4)

    return draw_img, arabic_text