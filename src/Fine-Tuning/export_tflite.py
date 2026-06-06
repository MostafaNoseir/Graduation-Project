from ultralytics import YOLO

# Load your best final model
model = YOLO(r'best.pt')

print("Exporting to TFLite (INT8)... This may take a few minutes.")

model.export(
    format='tflite',
    imgsz=640,
    int8=True,           # Best for mobile (smaller + faster)
    batch=1,
    device='cpu'         # Must be CPU for TFLite export
)

print("Export completed!")
print("TFLite model saved in the same folder as best.pt")