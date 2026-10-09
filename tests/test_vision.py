import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.density import (
    Detection,
    calculate_occupancy,
    density_level,
    density_levels_with_hysteresis,
    filter_in_roi,
)
from src.preprocessing import adaptive_canny, apply_clahe, suggest_road_roi
from src.roi import load_roi, roi_mask, save_roi


class VisionUtilitiesTest(unittest.TestCase):
    def test_roi_round_trip_and_detection_filter(self) -> None:
        points = [(20, 20), (80, 20), (90, 90), (10, 90)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "roi.json"
            save_roi(path, points, 100, 100)
            self.assertEqual(load_roi(path, 100, 100), points)

        detections = [
            Detection("car", 0.9, 30, 30, 50, 50),
            Detection("car", 0.8, 0, 0, 5, 5),
        ]
        inside = filter_in_roi(detections, points)
        self.assertEqual(len(inside), 1)
        mask = roi_mask(100, 100, points)
        occupancy = calculate_occupancy(100, 100, mask, inside)
        self.assertGreater(occupancy, 0)
        self.assertLessEqual(occupancy, 1)

    def test_density_thresholds(self) -> None:
        self.assertEqual(density_level(0.05, 0.1, 0.3), "Thấp")
        self.assertEqual(density_level(0.2, 0.1, 0.3), "Trung bình")
        self.assertEqual(density_level(0.4, 0.1, 0.3), "Cao")
        self.assertEqual(
            density_levels_with_hysteresis([0.05, 0.12, 0.09, 0.07], 0.1, 0.3, 0.02),
            ["Thấp", "Trung bình", "Trung bình", "Thấp"],
        )

    def test_self_intersecting_roi_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tự cắt|không hợp lệ"):
            roi_mask(100, 100, [(10, 10), (90, 90), (10, 90), (90, 10)])

    def test_preprocessing_produces_valid_images_and_roi(self) -> None:
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        frame[:, 100:] = 180
        enhanced = apply_clahe(frame)
        edges, low, high = adaptive_canny(enhanced)
        points = suggest_road_roi(enhanced)

        self.assertEqual(enhanced.shape, frame.shape)
        self.assertEqual(edges.shape, frame.shape[:2])
        self.assertLess(low, high)
        self.assertEqual(len(points), 4)
        roi_mask(200, 120, points)


if __name__ == "__main__":
    unittest.main()
