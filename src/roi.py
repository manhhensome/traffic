from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def validate_points(points: list[tuple[int, int]], width: int, height: int) -> None:
    if len(points) < 3:
        raise ValueError("ROI cần ít nhất ba điểm.")
    polygon = np.asarray(points, dtype=np.int32)
    if np.any(polygon[:, 0] < 0) or np.any(polygon[:, 0] >= width):
        raise ValueError("Tọa độ X của ROI nằm ngoài ảnh.")
    if np.any(polygon[:, 1] < 0) or np.any(polygon[:, 1] >= height):
        raise ValueError("Tọa độ Y của ROI nằm ngoài ảnh.")
    if abs(cv2.contourArea(polygon)) < width * height * 0.01:
        raise ValueError("ROI quá nhỏ hoặc không hợp lệ.")
    if len(points) == 4:
        def orientation(a, b, c):
            return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])

        def intersects(a, b, c, d):
            return orientation(a, b, c) * orientation(a, b, d) < 0 and orientation(c, d, a) * orientation(c, d, b) < 0

        if intersects(points[0], points[1], points[2], points[3]) or intersects(
            points[1], points[2], points[3], points[0]
        ):
            raise ValueError("Các cạnh ROI tự cắt nhau. Hãy sắp xếp các đỉnh theo vòng quanh vùng đường.")


def roi_mask(width: int, height: int, points: list[tuple[int, int]]) -> np.ndarray:
    validate_points(points, width, height)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    return mask


def overlay_roi(frame: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    output = frame.copy()
    polygon = np.asarray(points, dtype=np.int32)
    layer = output.copy()
    cv2.fillPoly(layer, [polygon], (30, 160, 255))
    cv2.addWeighted(layer, 0.18, output, 0.82, 0, output)
    cv2.polylines(output, [polygon], True, (0, 220, 255), 3)
    for index, point in enumerate(points, start=1):
        cv2.circle(output, point, 6, (0, 0, 255), -1)
        cv2.putText(output, str(index), point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return output


def save_roi(path: str | Path, points: list[tuple[int, int]], width: int, height: int) -> None:
    validate_points(points, width, height)
    normalized = [[x / width, y / height] for x, y in points]
    payload = {
        "schema_version": 1,
        "reference_width": width,
        "reference_height": height,
        "coordinate_system": "normalized",
        "points": normalized,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_roi(path: str | Path, width: int, height: int) -> list[tuple[int, int]] | None:
    roi_path = Path(path)
    if not roi_path.is_file():
        return None
    payload = json.loads(roi_path.read_text(encoding="utf-8"))
    points = [
        (min(width - 1, max(0, round(x * width))), min(height - 1, max(0, round(y * height))))
        for x, y in payload["points"]
    ]
    validate_points(points, width, height)
    return points

