from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_experiment_reports(data_directory: str | Path, reports_directory: str | Path) -> tuple[Path, Path]:
    data_path = Path(data_directory)
    reports_path = Path(reports_directory)
    reports_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for video_directory in sorted(data_path.glob("video-*")):
        metadata_path = video_directory / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for run_directory in sorted((video_directory / "runs").glob("run-*")):
            summary_path = run_directory / "summary.json"
            config_path = run_directory / "config.json"
            if not summary_path.is_file() or not config_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "completed":
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            suggestion_meta_path = run_directory / "roi_suggestion_meta.json"
            suggestion_meta = (
                json.loads(suggestion_meta_path.read_text(encoding="utf-8"))
                if suggestion_meta_path.is_file()
                else {}
            )
            evaluation_path = run_directory / "evaluation.json"
            evaluation = (
                json.loads(evaluation_path.read_text(encoding="utf-8"))
                if evaluation_path.is_file()
                else {}
            )
            mae = evaluation.get("count_mae") or {}
            rows.append(
                {
                    "video_id": metadata.get("video_id"),
                    "run_id": run_directory.name,
                    "intersection": metadata.get("intersection"),
                    "camera_id": metadata.get("camera_id"),
                    "lighting": metadata.get("lighting", "unknown"),
                    "weather": metadata.get("weather", "unknown"),
                    "method": config.get("method"),
                    "use_clahe": config.get("use_clahe", False),
                    "roi_edge_method": suggestion_meta.get("method"),
                    "roi_manual_edit_seconds": suggestion_meta.get("manual_edit_seconds"),
                    "frame_stride": config.get("frame_stride"),
                    "image_size": config.get("image_size"),
                    "processing_fps": summary.get("processing_fps"),
                    "pipeline_fps": summary.get("pipeline_fps"),
                    "latency_p95_ms": summary.get("latency_p95_ms"),
                    "mean_count": summary.get("mean_vehicle_or_object_count"),
                    "mean_occupancy": summary.get("mean_occupancy"),
                    "mae_total": mae.get("n_total"),
                    "map50": evaluation.get("mAP50"),
                    "density_macro_f1": evaluation.get("density_macro_f1"),
                    "roi_iou": evaluation.get("roi_iou"),
                }
            )

    columns = [
        "video_id", "run_id", "intersection", "camera_id", "lighting", "weather",
        "method", "use_clahe", "roi_edge_method", "roi_manual_edit_seconds",
        "frame_stride", "image_size", "processing_fps",
        "pipeline_fps", "latency_p95_ms", "mean_count", "mean_occupancy",
        "mae_total", "map50", "density_macro_f1", "roi_iou",
    ]
    details = pd.DataFrame(rows, columns=columns)
    detail_path = reports_path / "experiment_runs.csv"
    details.to_csv(detail_path, index=False)

    aggregate_path = reports_path / "condition_summary.csv"
    if details.empty:
        pd.DataFrame().to_csv(aggregate_path, index=False)
    else:
        numeric = [
            "processing_fps", "pipeline_fps", "latency_p95_ms", "mean_count",
            "mean_occupancy", "mae_total", "map50", "density_macro_f1", "roi_iou",
        ]
        aggregate = details.groupby(
            ["lighting", "method", "use_clahe"], dropna=False
        )[numeric].mean(numeric_only=True).reset_index()
        aggregate["n_runs"] = details.groupby(
            ["lighting", "method", "use_clahe"], dropna=False
        ).size().to_numpy()
        aggregate.to_csv(aggregate_path, index=False)
    return detail_path, aggregate_path
