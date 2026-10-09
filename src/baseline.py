from __future__ import annotations

import cv2
import numpy as np

from src.density import Detection


class MOG2Detector:
    def __init__(self, frame_area: int, min_area_ratio: float = 0.0005) -> None:
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=24,
            detectShadows=True,
        )
        self.minimum_area = max(80, int(frame_area * min_area_ratio))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process(self, frame: np.ndarray, roi_mask: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        foreground = self.subtractor.apply(frame)
        # MOG2 uses 127 for shadows and 255 for foreground.
        foreground = np.where(foreground == 255, 255, 0).astype(np.uint8)
        foreground = cv2.bitwise_and(foreground, roi_mask)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, self.kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            if cv2.contourArea(contour) < self.minimum_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            detections.append(Detection("moving_object", 1.0, x, y, x + width, y + height))
        return detections, foreground

