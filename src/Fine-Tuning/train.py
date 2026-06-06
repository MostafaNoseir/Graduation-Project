from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo26n.pt')   # or full path if needed

    model.train(
        data=r'D:\FCDS\Graduation Project\object detection\Data\combined_dataset\data.yaml',
        epochs=100,
        imgsz=640,
        batch=32,                    # Safe starting value for RTX 3070 Laptop
        device=0,
        patience=30,
        lr0=0.001,
        lrf=0.01,
        workers=4,
        cache=False,                # 'ram' or 'disk'
        seed=42,
        val=True,
        plots=True,
        name='yolo26n_coco_plus_40new_v1',

        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,

        freeze=22,                   # Good choice for preserving COCO
    )

    print("Training completed!")