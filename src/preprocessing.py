from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def adaptive_canny(frame: np.ndarray, percentile: float = 85.0) -> tuple[np.ndarray, int, int]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    nonzero = magnitude[magnitude > 0]
    high = int(np.percentile(nonzero, percentile)) if nonzero.size else 100
    high = int(np.clip(high, 30, 255))
    low = max(10, int(0.4 * high))
    return cv2.Canny(blurred, low, high, L2gradient=True), low, high


def fixed_canny(frame: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, low, high, L2gradient=True)


def sobel_edges(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def suggest_road_roi(
    frame: np.ndarray,
    edge_mode: str = "adaptive",
    fixed_thresholds: tuple[int, int] = (100, 200),
) -> list[tuple[int, int]]:
    """Suggest a road trapezoid from long lower-frame edges, with a safe fallback."""
    height, width = frame.shape[:2]
    if edge_mode == "fixed":
        edges = fixed_canny(frame, *fixed_thresholds)
    else:
        edges, _, _ = adaptive_canny(frame)
    edges[: int(height * 0.35), :] = 0
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(35, width // 30),
        minLineLength=max(40, width // 8),
        maxLineGap=max(20, width // 40),
    )
    top_y = int(height * 0.48)
    bottom_y = int(height * 0.95)
    left_candidates: list[tuple[float, float]] = []
    right_candidates: list[tuple[float, float]] = []

    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue
            intercept = y1 - slope * x1
            x_top = (top_y - intercept) / slope
            x_bottom = (bottom_y - intercept) / slope
            if slope < 0 and x_bottom < width * 0.65:
                left_candidates.append((x_top, x_bottom))
            elif slope > 0 and x_bottom > width * 0.35:
                right_candidates.append((x_top, x_bottom))

    if left_candidates and right_candidates:
        left_top, left_bottom = np.median(left_candidates, axis=0)
        right_top, right_bottom = np.median(right_candidates, axis=0)
        points = [
            (int(np.clip(left_top, 0, width - 1)), top_y),
            (int(np.clip(right_top, 0, width - 1)), top_y),
            (int(np.clip(right_bottom, 0, width - 1)), bottom_y),
            (int(np.clip(left_bottom, 0, width - 1)), bottom_y),
        ]
        if points[0][0] + 20 < points[1][0] and points[3][0] + 20 < points[2][0]:
            return points

    return [
        (int(width * 0.34), top_y),
        (int(width * 0.66), top_y),
        (int(width * 0.92), bottom_y),
        (int(width * 0.08), bottom_y),
    ]

