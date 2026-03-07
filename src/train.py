from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    
    model.train(
        data="configs/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,        # prueba 8 primero
        workers=0,
        amp=True,
        cache="disk",   # usa "ram" si tienes RAM de sobra
        patience=12,
        device=0,
        project="runs_local",
        name="ppe_rtx2060"
    )