from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 cho bốn lớp phương tiện.")
    parser.add_argument("--data", required=True, help="Đường dẫn dataset.yaml")
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        project="models/training",
        name="hanoi-traffic",
    )


if __name__ == "__main__":
    main()
