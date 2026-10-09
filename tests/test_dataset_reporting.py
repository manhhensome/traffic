import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.dataset import create_grouped_splits
from src.reporting import build_experiment_reports


class DatasetAndReportingTest(unittest.TestCase):
    def test_grouped_split_keeps_session_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata = [
                {"video_id": 1, "recording_session_id": "a"},
                {"video_id": 2, "recording_session_id": "a"},
                {"video_id": 3, "recording_session_id": "b"},
                {"video_id": 4, "recording_session_id": "c"},
                {"video_id": 5, "recording_session_id": "d"},
            ]
            assignments = create_grouped_splits(temporary_directory, metadata)
            self.assertEqual(assignments["1"], assignments["2"])
            self.assertTrue((Path(temporary_directory) / "splits.json").is_file())

    def test_report_collects_completed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video = root / "data" / "video-1"
            run = video / "runs" / "run-0001"
            run.mkdir(parents=True)
            (video / "metadata.json").write_text(
                json.dumps({"video_id": 1, "lighting": "night", "weather": "clear"}),
                encoding="utf-8",
            )
            (run / "summary.json").write_text(
                json.dumps({"status": "completed", "processing_fps": 5, "mean_occupancy": 0.2}),
                encoding="utf-8",
            )
            (run / "config.json").write_text(
                json.dumps({"method": "yolo", "use_clahe": True}), encoding="utf-8"
            )
            (run / "evaluation.json").write_text(
                json.dumps({"mAP50": 0.5, "count_mae": {"n_total": 1.2}}), encoding="utf-8"
            )
            detail, aggregate = build_experiment_reports(root / "data", root / "reports")
            self.assertEqual(len(pd.read_csv(detail)), 1)
            self.assertEqual(len(pd.read_csv(aggregate)), 1)


if __name__ == "__main__":
    unittest.main()
