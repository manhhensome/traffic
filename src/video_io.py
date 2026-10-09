from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import pandas as pd


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def inspect_video(video_path: str | Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Không thể mở video. Hãy kiểm tra codec hoặc file bị hỏng.")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise ValueError("Video không có kích thước frame hợp lệ.")
        if fps <= 0:
            fps = 25.0
        duration = frame_count / fps if frame_count > 0 else 0.0
        return VideoInfo(width, height, fps, frame_count, duration)
    finally:
        capture.release()


def read_frame(video_path: str | Path, timestamp_seconds: float = 0.0):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Không thể mở video.")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_seconds) * 1000)
        success, frame = capture.read()
        if not success or frame is None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = capture.read()
        if not success or frame is None:
            raise ValueError("Không đọc được frame từ video.")
        return frame
    finally:
        capture.release()


def extract_frames(
    video_path: str | Path,
    output_directory: str | Path,
    interval_seconds: float = 1.0,
) -> Path:
    if interval_seconds <= 0:
        raise ValueError("Khoảng trích frame phải lớn hơn 0 giây.")
    info = inspect_video(video_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    rows: list[dict] = []
    timestamp = 0.0
    try:
        while timestamp <= info.duration_seconds + 1e-6:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, frame = capture.read()
            if not success or frame is None:
                break
            frame_index = int(round(timestamp * info.fps))
            filename = f"frame_{frame_index:08d}.jpg"
            if not cv2.imwrite(str(output / filename), frame):
                raise OSError(f"Không ghi được {filename}")
            rows.append(
                {
                    "filename": filename,
                    "frame_index": frame_index,
                    "timestamp_s": round(timestamp, 3),
                }
            )
            timestamp += interval_seconds
    finally:
        capture.release()
    manifest = output / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest

