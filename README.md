# YOLOv11n TFLite – Offline Detection (Ready for Mobile Application)

## What's Included
- `yolo11n_float32.tflite`
- `labels.txt` → 80 COCO classes (0 person, 1 bicycle, etc.)
- `yolo_detector.py` → Full working Python reference
- `yolo_detector_API.py` → working Python reference version for API
- `web_API.py` → run API interface

## How to Use on Mobile

### Input
- Image size: Any (will be resized with letterbox)
- Format: RGB or BGR → will be converted
- Padding: Gray (114,114,114)

### Preprocessing (Must Match Exactly!)
```python
Resize + letterbox to 640x640
Pad with 114
Divide by 255.0 → float32

### Prediction
- Detect each object at the image
