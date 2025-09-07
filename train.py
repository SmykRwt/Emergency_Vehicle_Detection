from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Or yolov8s.pt, etc.
model.train(data="vehicles.yaml", epochs=30, imgsz=2626)
