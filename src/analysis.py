from __future__ import annotations

import json
import hashlib
import platform
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from src.baseline import MOG2Detector
from src.density import (
    Detection,
    calculate_occupancy,
    count_classes,
    density_level,
    density_levels_with_hysteresis,
    filter_in_roi,
)
from src.evaluation import (
    evaluate_counts,
    evaluate_density_levels,
    evaluate_map50,
    evaluate_roi_iou,
)
from src.preprocessing import apply_clahe
from src.roi import roi_mask, save_roi
from src.video_io import inspect_video


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class AnalysisConfig:
    method: str = "yolo"
    model_path: str = "models/yolov8n.pt"
    device: str = "cpu"
    confidence: float = 0.35
    image_size: int = 640
    frame_stride: int = 5
    max_seconds: float = 0.0
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    medium_threshold: float = 0.12
    high_threshold: float = 0.28
    hysteresis_margin: float = 0.02
    mog2_warmup_seconds: float = 2.0


def _next_run_directory(video_directory: Path) -> Path:
    runs_directory = video_directory / "runs"
    runs_directory.mkdir(exist_ok=True)
    existing = []
    for child in runs_directory.glob("run-*"):
        try:
            existing.append(int(child.name.split("-")[-1]))
        except ValueError:
            continue
    run_directory = runs_directory / f"run-{max(existing, default=0) + 1:04d}"
    run_directory.mkdir()
    return run_directory


def _draw_result(
    frame: np.ndarray,
    roi_points: list[tuple[int, int]],
    detections: list[Detection],
    occupancy: float,
    level: str,
    method: str,
) -> np.ndarray:
    output = frame.copy()
    polygon = np.asarray(roi_points, dtype=np.int32)
    cv2.polylines(output, [polygon], True, (0, 220, 255), 3)
    colors = {
        "motorcycle": (0, 200, 255),
        "car": (50, 220, 50),
        "bus": (255, 120, 20),
        "truck": (220, 50, 220),
        "moving_object": (0, 180, 255),
    }
    for detection in detections:
        color = colors.get(detection.class_name, (255, 255, 255))
        cv2.rectangle(output, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
        label = detection.class_name
        if detection.confidence < 1:
            label += f" {detection.confidence:.2f}"
        cv2.putText(
            output,
            label,
            (detection.x1, max(20, detection.y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    text = f"{method.upper()} | vehicles/objects: {len(detections)} | occupancy: {occupancy:.1%} | {level}"
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 900), 40), (20, 20, 20), -1)
    cv2.putText(output, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return output


def _write_run_status(run_directory: Path, status: str, error: str | None = None) -> None:
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    (run_directory / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (run_directory / "run.log").open("a", encoding="utf-8") as log:
        log.write(f"{payload['updated_at']}\t{status}\t{error or ''}\n")


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_analysis_impl(
    video_path: str | Path,
    video_directory: str | Path,
    roi_points: list[tuple[int, int]],
    config: AnalysisConfig,
    run_directory: Path,
    progress: ProgressCallback | None = None,
) -> Path:
    video_path = Path(video_path)
    video_directory = Path(video_directory)
    info = inspect_video(video_path)
    mask = roi_mask(info.width, info.height, roi_points)
    (run_directory / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        import yaml

        (run_directory / "config.yaml").write_text(
            yaml.safe_dump(asdict(config), allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    except ImportError:
        pass
    try:
        import cv2 as cv2_package
        import numpy as numpy_package
        import pandas as pandas_package
        import ultralytics

        dependency_versions = {
            "opencv": cv2_package.__version__,
            "numpy": numpy_package.__version__,
            "pandas": pandas_package.__version__,
            "ultralytics": ultralytics.__version__,
        }
    except ImportError:
        dependency_versions = {}
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependency_versions,
        "model_sha256": _file_sha256(Path(config.model_path)) if config.method == "yolo" else None,
        "class_mapping": ["motorcycle", "car", "bus", "truck"],
    }
    (run_directory / "runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_roi(run_directory / "roi.json", roi_points, info.width, info.height)
    cv2.imwrite(str(run_directory / "roi_mask.png"), mask)
    suggestion_meta_path = video_directory / "roi_suggestion_meta.json"
    if suggestion_meta_path.is_file():
        (run_directory / "roi_suggestion_meta.json").write_text(
            suggestion_meta_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    if config.method == "yolo":
        from src.detector import YOLOVehicleDetector

        detector = YOLOVehicleDetector(config.model_path, config.device)
        baseline = None
    elif config.method == "mog2":
        detector = None
        baseline = MOG2Detector(info.width * info.height)
    else:
        raise ValueError("Phương pháp phải là yolo hoặc mog2.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Không thể mở video để phân tích.")

    max_frames = info.frame_count
    if config.max_seconds > 0:
        max_frames = min(max_frames, int(config.max_seconds * info.fps))
    if max_frames <= 0:
        max_frames = int(config.max_seconds * info.fps) if config.max_seconds > 0 else 1

    output_fps = max(1.0, info.fps / max(1, config.frame_stride))
    writer = cv2.VideoWriter(
        str(run_directory / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Không tạo được video kết quả bằng codec mp4v.")

    rows: list[dict] = []
    detection_rows: list[dict] = []
    decoded_frames = 0
    read_failures = 0
    sampled_frames = 0
    sample_latencies_ms: list[float] = []
    start_time = time.perf_counter()

    try:
        while decoded_frames < max_frames:
            success, frame = capture.read()
            if not success or frame is None:
                if decoded_frames + 1 < max_frames:
                    read_failures += 1
                break
            frame_index = decoded_frames
            decoded_frames += 1
            decoded_timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            prepared = apply_clahe(frame, config.clahe_clip_limit) if config.use_clahe else frame

            sample_start = time.perf_counter()
            foreground_mask = None
            if baseline is not None:
                all_detections, foreground_mask = baseline.process(prepared, mask)
            elif frame_index % config.frame_stride == 0:
                all_detections = detector.predict(prepared, config.confidence, config.image_size)
            else:
                all_detections = []

            if frame_index % config.frame_stride != 0:
                continue

            detections = filter_in_roi(all_detections, roi_points)
            if foreground_mask is not None:
                roi_area = max(1, int(np.count_nonzero(mask)))
                occupancy = float(np.count_nonzero(foreground_mask) / roi_area)
                occupancy_method = "foreground_mask"
            else:
                occupancy = calculate_occupancy(info.width, info.height, mask, detections)
                occupancy_method = "bounding_box_union"
            counts = count_classes(detections)
            timestamp = (
                decoded_timestamp_ms / 1000
                if decoded_timestamp_ms > 0
                else frame_index / info.fps
            )
            sampled_frames += 1
            latency_ms = (time.perf_counter() - sample_start) * 1000
            sample_latencies_ms.append(latency_ms)
            is_valid = config.method != "mog2" or timestamp >= config.mog2_warmup_seconds
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": round(timestamp, 3),
                    "n_motorcycle": counts["motorcycle"],
                    "n_car": counts["car"],
                    "n_bus": counts["bus"],
                    "n_truck": counts["truck"],
                    "n_total": counts["total"],
                    "occupancy": occupancy,
                    "occupancy_method": occupancy_method,
                    "foreground_pixels": int(np.count_nonzero(foreground_mask))
                    if foreground_mask is not None
                    else None,
                    "latency_ms": latency_ms,
                    "valid": is_valid,
                }
            )
            for detection in detections:
                detection_rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_s": round(timestamp, 3),
                        "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        "x1": detection.x1,
                        "y1": detection.y1,
                        "x2": detection.x2,
                        "y2": detection.y2,
                    }
                )

            raw_level = density_level(
                occupancy, config.medium_threshold, config.high_threshold
            )
            writer.write(_draw_result(frame, roi_points, detections, occupancy, raw_level, config.method))
            if progress and sampled_frames % 5 == 0:
                progress(
                    min(1.0, decoded_frames / max_frames),
                    f"Đã xử lý {decoded_frames}/{max_frames} frame",
                )
    finally:
        capture.release()
        writer.release()

    if not rows:
        raise RuntimeError("Không có frame hợp lệ để tạo kết quả.")

    elapsed = time.perf_counter() - start_time
    frame_table = pd.DataFrame(rows)
    frame_table.to_csv(run_directory / "frame_metrics.csv", index=False)
    frame_table["second"] = np.floor(frame_table["timestamp_s"]).astype(int)
    table = frame_table.groupby("second", as_index=False).agg(
        t_start_s=("timestamp_s", "min"),
        t_end_s=("timestamp_s", "max"),
        n_samples=("frame_index", "count"),
        n_motorcycle=("n_motorcycle", "mean"),
        n_car=("n_car", "mean"),
        n_bus=("n_bus", "mean"),
        n_truck=("n_truck", "mean"),
        n_total=("n_total", "mean"),
        occupancy=("occupancy", "mean"),
        valid_fraction=("valid", "mean"),
        latency_median_ms=("latency_ms", "median"),
        latency_p95_ms=("latency_ms", lambda values: float(np.percentile(values, 95))),
        foreground_pixels_mean=("foreground_pixels", "mean"),
    )
    table["timestamp_s"] = table["second"].astype(float)
    table["occupancy_smooth"] = table["occupancy"].rolling(5, min_periods=1).mean()
    table["density_level"] = density_levels_with_hysteresis(
        table["occupancy_smooth"].tolist(),
        config.medium_threshold,
        config.high_threshold,
        config.hysteresis_margin,
    )
    table.loc[table["valid_fraction"] == 0, "density_level"] = "Khởi tạo"
    for column in ["n_motorcycle", "n_car", "n_bus", "n_truck", "n_total"]:
        table[f"{column}_mean"] = table[column]
    table["occupancy_mean"] = table["occupancy"]
    timeseries_path = run_directory / "timeseries.csv"
    table.to_csv(timeseries_path, index=False)
    pd.DataFrame(
        detection_rows,
        columns=["frame_index", "timestamp_s", "class_name", "confidence", "x1", "y1", "x2", "y2"],
    ).to_csv(run_directory / "detections.csv", index=False)

    valid_table = frame_table[frame_table["valid"]]
    if valid_table.empty:
        valid_table = frame_table
    latency_array = np.asarray(sample_latencies_ms, dtype=float)
    summary = {
        "status": "completed",
        "method": config.method,
        "sampled_frames": sampled_frames,
        "decoded_frames": decoded_frames,
        "elapsed_seconds": elapsed,
        "processing_fps": sampled_frames / elapsed if elapsed else None,
        "pipeline_fps": decoded_frames / elapsed if elapsed else None,
        "read_failures": read_failures,
        "latency_median_ms": float(np.median(latency_array)),
        "latency_p95_ms": float(np.percentile(latency_array, 95)),
        "latency_std_ms": float(np.std(latency_array)),
        "sample_fps_std": float(np.std(1000.0 / np.maximum(latency_array, 1e-9))),
        "sample_fps_cv": float(
            np.std(1000.0 / np.maximum(latency_array, 1e-9))
            / np.mean(1000.0 / np.maximum(latency_array, 1e-9))
        ),
        "mean_vehicle_or_object_count": float(valid_table["n_total"].mean()),
        "max_vehicle_or_object_count": int(valid_table["n_total"].max()),
        "mean_occupancy": float(valid_table["occupancy"].mean()),
        "max_occupancy": float(valid_table["occupancy"].max()),
        "occupancy_method": str(frame_table["occupancy_method"].iloc[0]),
        "warmup_seconds": config.mog2_warmup_seconds if config.method == "mog2" else 0,
        "density_distribution": table["density_level"].value_counts().to_dict(),
    }
    (run_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    evaluation_path = run_directory / "evaluation.json"
    evaluate_counts(
        timeseries_path,
        video_directory / "annotations" / "counts.csv",
        evaluation_path,
        tolerance_seconds=max(0.6, config.frame_stride / info.fps),
    )
    if config.method == "yolo":
        evaluate_map50(
            config.model_path,
            video_directory / "annotations" / "dataset.yaml",
            evaluation_path,
            image_size=config.image_size,
            device=config.device,
        )
    else:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation["mAP50"] = None
        evaluation["mAP50_note"] = "N/A: MOG2 không phân loại phương tiện."
        evaluation_path.write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    evaluate_density_levels(
        timeseries_path,
        video_directory / "annotations" / "density.csv",
        evaluation_path,
    )
    evaluate_roi_iou(
        video_directory / "roi_suggestion.json",
        video_directory / "annotations" / "roi_reference.json",
        info.width,
        info.height,
        evaluation_path,
    )
    if progress:
        progress(1.0, "Hoàn tất phân tích")
    return run_directory


def run_analysis(
    video_path: str | Path,
    video_directory: str | Path,
    roi_points: list[tuple[int, int]],
    config: AnalysisConfig,
    progress: ProgressCallback | None = None,
) -> Path:
    video_directory = Path(video_directory)
    run_directory = _next_run_directory(video_directory)
    _write_run_status(run_directory, "queued")
    _write_run_status(run_directory, "processing")
    try:
        result = _run_analysis_impl(
            video_path,
            video_directory,
            roi_points,
            config,
            run_directory,
            progress,
        )
    except KeyboardInterrupt:
        _write_run_status(run_directory, "interrupted", "Người dùng đã ngắt tác vụ")
        raise
    except Exception as error:
        _write_run_status(run_directory, "failed", str(error))
        (run_directory / "summary.json").write_text(
            json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise
    _write_run_status(run_directory, "completed")
    return result


def recover_incomplete_runs(data_directory: str | Path) -> int:
    recovered = 0
    for status_path in Path(data_directory).glob("video-*/runs/run-*/status.json"):
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") in {"queued", "processing"}:
            _write_run_status(status_path.parent, "interrupted", "Ứng dụng đã dừng trước khi run hoàn tất")
            recovered += 1
    return recovered


def list_runs(video_directory: str | Path) -> list[Path]:
    runs_directory = Path(video_directory) / "runs"
    if not runs_directory.is_dir():
        return []
    return sorted(
        [
            path
            for path in runs_directory.glob("run-*")
            if (path / "summary.json").is_file()
            and json.loads((path / "summary.json").read_text(encoding="utf-8")).get("status") == "completed"
        ],
        reverse=True,
    )
