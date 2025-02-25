if __name__ == '__main__':
    import torch
    from ultralytics import YOLO

    model = YOLO("yolo11n-seg.pt")
    train_results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=256,
        device="cuda"
    )
