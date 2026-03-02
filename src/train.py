from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    
    model.train(
        data = "configs/data.yaml",
        epochs = 5,
        imgsz = 640,
        batch = 16,
        fraction = 0.05,
        device = "mps",
        project = "runs",
        name = "debug-small-mps"
    )