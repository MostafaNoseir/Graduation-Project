from ultralytics import YOLO

if __name__ == '__main__':
    
    # === RESUME TRAINING IN THE SAME FOLDER ===
    last_pt = r'runs/detect/yolo26n_coco_plus_40new_v17/weights/last.pt'
    
    model = YOLO(last_pt)

    model.train(
        resume=True,                    # Must be True
        data=r'D:\FCDS\Graduation Project\object detection\Data\combined_dataset\data.yaml',
        
        # Do NOT put 'name=' here when using resume=True
        # Ultralytics will automatically use the original run name
        
        epochs=100,
        imgsz=640,
        batch=16,                       # Keep same as your original training
        device=0,
        patience=30,
        lr0=0.001,
        lrf=0.01,
        workers=4,
        cache=False,
        seed=42,
        val=True,
        plots=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        freeze=22,
    )

    print("Training resumed successfully in the original run folder!")
    
# conda activate torch_gpu
# D:
# cd "D:\FCDS\Graduation Project\object detection\Data"
# python resume_train.py