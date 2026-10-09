from __future__ import annotations

from pathlib import Path

import numpy as np

from src.density import Detection, VEHICLE_CLASSES


class YOLOVehicleDetector:
    def __init__(self, model_path: str | Path = "models/yolov8n.pt", device: str = "cpu") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Chưa cài ultralytics. Chạy: pip install -r requirements.txt"
            ) from error
        self.model = YOLO(str(model_path))
        self.device = device
        names = self.model.names
        self.names = dict(enumerate(names)) if isinstance(names, list) else dict(names)
        self.vehicle_ids = [
            class_id for class_id, name in self.names.items() if name in VEHICLE_CLASSES
        ]
        if not self.vehicle_ids:
            raise ValueError("Mô hình không có lớp motorcycle/car/bus/truck.")

    def predict(self, frame: np.ndarray, confidence: float, image_size: int) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=confidence,
            imgsz=image_size,
            classes=self.vehicle_ids,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return detections
        coordinates = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        for box, score, class_id in zip(coordinates, confidences, class_ids):
            x1, y1, x2, y2 = (int(value) for value in box)
            detections.append(
                Detection(self.names[class_id], float(score), x1, y1, x2, y2)
            )
        return detections

