from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bottom_center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, self.y2)


VEHICLE_CLASSES = ("motorcycle", "car", "bus", "truck")


def filter_in_roi(detections: list[Detection], roi_points: list[tuple[int, int]]) -> list[Detection]:
    polygon = np.asarray(roi_points, dtype=np.float32)
    return [
        detection
        for detection in detections
        if cv2.pointPolygonTest(polygon, detection.bottom_center, False) >= 0
    ]


def calculate_occupancy(
    width: int,
    height: int,
    roi: np.ndarray,
    detections: list[Detection],
) -> float:
    roi_area = int(np.count_nonzero(roi))
    if roi_area == 0:
        raise ValueError("ROI rỗng.")
    occupied = np.zeros((height, width), dtype=np.uint8)
    for detection in detections:
        x1 = int(np.clip(detection.x1, 0, width - 1))
        y1 = int(np.clip(detection.y1, 0, height - 1))
        x2 = int(np.clip(detection.x2, 0, width - 1))
        y2 = int(np.clip(detection.y2, 0, height - 1))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(occupied, (x1, y1), (x2, y2), 255, -1)
    occupied = cv2.bitwise_and(occupied, roi)
    return min(1.0, np.count_nonzero(occupied) / roi_area)


def density_level(occupancy: float, medium_threshold: float, high_threshold: float) -> str:
    if occupancy >= high_threshold:
        return "Cao"
    if occupancy >= medium_threshold:
        return "Trung bình"
    return "Thấp"


def density_levels_with_hysteresis(
    occupancies: list[float],
    medium_threshold: float,
    high_threshold: float,
    margin: float = 0.02,
) -> list[str]:
    """Keep a density state until occupancy crosses a return threshold."""
    levels: list[str] = []
    current = "Thấp"
    for occupancy in occupancies:
        if current == "Thấp":
            if occupancy >= high_threshold:
                current = "Cao"
            elif occupancy >= medium_threshold:
                current = "Trung bình"
        elif current == "Trung bình":
            if occupancy >= high_threshold:
                current = "Cao"
            elif occupancy < max(0.0, medium_threshold - margin):
                current = "Thấp"
        else:
            if occupancy < max(0.0, high_threshold - margin):
                current = (
                    "Trung bình"
                    if occupancy >= max(0.0, medium_threshold - margin)
                    else "Thấp"
                )
        levels.append(current)
    return levels


def count_classes(detections: list[Detection]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in VEHICLE_CLASSES}
    for detection in detections:
        if detection.class_name in counts:
            counts[detection.class_name] += 1
    counts["total"] = len(detections)
    return counts
