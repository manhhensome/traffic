import json
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil

from src.analysis import AnalysisConfig, run_analysis
from src.jobs import read_job, start_analysis_job
from src.video_io import extract_frames


class AnalysisPipelineTest(unittest.TestCase):
    def test_mog2_pipeline_writes_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_directory = root / "video-1"
            video_directory.mkdir()
            video_path = video_directory / "original.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
            )
            self.assertTrue(writer.isOpened())
            for frame_index in range(20):
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                x = 10 + frame_index * 4
                cv2.rectangle(frame, (x, 55), (x + 25, 90), (255, 255, 255), -1)
                writer.write(frame)
            writer.release()

            manifest = extract_frames(video_path, video_directory / "frames", interval_seconds=0.5)
            self.assertTrue(manifest.is_file())
            self.assertGreaterEqual(len(pd.read_csv(manifest)), 4)

            run = run_analysis(
                video_path,
                video_directory,
                [(5, 35), (155, 35), (155, 115), (5, 115)],
                AnalysisConfig(method="mog2", frame_stride=2, max_seconds=0),
            )

            expected = [
                "annotated.mp4",
                "config.json",
                "config.yaml",
                "detections.csv",
                "evaluation.json",
                "frame_metrics.csv",
                "roi.json",
                "roi_mask.png",
                "run.log",
                "runtime.json",
                "status.json",
                "summary.json",
                "timeseries.csv",
            ]
            for filename in expected:
                self.assertTrue((run / filename).is_file(), filename)
            table = pd.read_csv(run / "timeseries.csv")
            self.assertEqual(len(table), 2)
            self.assertEqual(len(pd.read_csv(run / "frame_metrics.csv")), 10)
            self.assertTrue(table["occupancy"].between(0, 1).all())
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["method"], "mog2")
            status = json.loads((run / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "completed")

    def test_background_worker_completes_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_directory = root / "video-1"
            video_directory.mkdir()
            video_path = video_directory / "original.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (80, 60)
            )
            for frame_index in range(8):
                frame = np.zeros((60, 80, 3), dtype=np.uint8)
                cv2.rectangle(frame, (5 + frame_index * 3, 30), (20 + frame_index * 3, 50), (255, 255, 255), -1)
                writer.write(frame)
            writer.release()

            project = Path(__file__).parents[1]
            job_path = start_analysis_job(
                project,
                video_path,
                video_directory,
                [(2, 15), (77, 15), (77, 58), (2, 58)],
                AnalysisConfig(method="mog2", frame_stride=2),
            )
            deadline = time.time() + 15
            payload = read_job(job_path)
            while payload["status"] in {"queued", "processing"} and time.time() < deadline:
                time.sleep(0.1)
                payload = read_job(job_path)
            self.assertEqual(payload["status"], "completed", payload.get("error"))
            self.assertTrue(Path(payload["run_directory"]).is_dir())
            pid = int(payload["pid"])
            while psutil.pid_exists(pid) and time.time() < deadline:
                time.sleep(0.05)
            read_job(job_path)


if __name__ == "__main__":
    unittest.main()
