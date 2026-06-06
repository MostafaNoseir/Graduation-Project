from ultralytics import YOLO
from pathlib import Path

# ================== CONFIG ==================
# Choose which checkpoint to test:
model_path = r'best.pt'   # Latest checkpoint
# model_path = r'runs/detect/yolo26n_coco_plus_40new_v1/weights/best.pt'  # Best so far

# Path to your test image
image_path = "bus.jpg"   # ← CHANGE THIS TO YOUR IMAGE PATH

# Confidence threshold
CONF_THRESHOLD = 0.2

# ================== Load Model & Run Inference ==================
model = YOLO(model_path)

print(f"Testing image: {image_path}")
print(f"Using model: {model_path}\n")

results = model(
    image_path,
    conf=CONF_THRESHOLD,
    iou=0.45,
    save=True,           # Saves image with boxes in runs/detect/predict/
    show=False
)

# ================== Print Results ==================
result = results[0]  # First image result

print("═" * 80)
print("DETECTION RESULTS")
print("═" * 80)

detected_objects = []

for box in result.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    class_name = result.names[cls_id]
    
    detected_objects.append(f"{class_name} ({conf:.3f})")

    # Optional: print bounding box coordinates
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"→ {class_name:20} | Confidence: {conf:.3f} | Box: ({int(x1)},{int(y1)}) - ({int(x2)},{int(y2)})")

if detected_objects:
    print(f"\n✅ Detected {len(detected_objects)} objects:")
    for obj in detected_objects:
        print(f"   • {obj}")
else:
    print("❌ No objects detected above confidence threshold.")

print(f"\nResult image saved to: runs/detect/predict/")