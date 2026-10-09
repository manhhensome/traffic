from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.roi import load_roi, roi_mask


COUNT_COLUMNS = ["n_total", "n_motorcycle", "n_car", "n_bus", "n_truck"]


def evaluate_counts(
    prediction_path: str | Path,
    ground_truth_path: str | Path,
    output_path: str | Path,
    tolerance_seconds: float = 0.6,
) -> dict:
    ground_truth_file = Path(ground_truth_path)
    result: dict = {
        "status": "not_evaluated",
        "reason": "Chưa có annotations/counts.csv",
        "mAP50": None,
        "count_mae": None,
    }
    if not ground_truth_file.is_file():
        Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    predictions = pd.read_csv(prediction_path)
    ground_truth = pd.read_csv(ground_truth_file)
    if "timestamp_s" not in ground_truth.columns or "timestamp_s" not in predictions.columns:
        result["reason"] = "counts.csv cần cột timestamp_s"
        Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    errors: dict[str, list[float]] = {column: [] for column in COUNT_COLUMNS}
    matched = 0
    prediction_times = predictions["timestamp_s"].to_numpy()
    for _, truth in ground_truth.iterrows():
        if prediction_times.size == 0:
            break
        index = int(np.argmin(np.abs(prediction_times - float(truth["timestamp_s"]))))
        if abs(prediction_times[index] - float(truth["timestamp_s"])) > tolerance_seconds:
            continue
        prediction = predictions.iloc[index]
        matched += 1
        for column in COUNT_COLUMNS:
            if column in ground_truth.columns and column in predictions.columns:
                errors[column].append(abs(float(prediction[column]) - float(truth[column])))

    mae = {
        column: float(np.mean(values))
        for column, values in errors.items()
        if values
    }
    result = {
        "status": "evaluated" if matched else "not_evaluated",
        "reason": None if matched else "Không ghép được timestamp giữa dự đoán và nhãn",
        "matched_samples": matched,
        "count_mae": mae or None,
        "mAP50": None,
        "mAP50_note": "Cần bộ nhãn bounding box YOLO và data.yaml để tính mAP@0.5.",
    }
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_map50(
    model_path: str | Path,
    dataset_yaml: str | Path,
    output_path: str | Path,
    image_size: int = 640,
    device: str = "cpu",
) -> dict:
    output_file = Path(output_path)
    result = json.loads(output_file.read_text(encoding="utf-8")) if output_file.is_file() else {}
    dataset_path = Path(dataset_yaml)
    if not dataset_path.is_file():
        result["mAP50"] = None
        result["mAP50_note"] = "Chưa có annotations/dataset.yaml và nhãn bounding box YOLO."
    else:
        try:
            from ultralytics import YOLO

            metrics = YOLO(str(model_path)).val(
                data=str(dataset_path),
                imgsz=image_size,
                device=device,
                verbose=False,
            )
            result["mAP50"] = float(metrics.box.map50)
            result["mAP50_95"] = float(metrics.box.map)
            class_names = getattr(metrics, "names", {})
            per_class_ap50 = getattr(metrics.box, "ap50", None)
            if per_class_ap50 is not None:
                result["mAP50_per_class"] = {
                    str(class_names.get(index, index)): float(value)
                    for index, value in enumerate(per_class_ap50)
                }
            result["mAP_scope"] = "full_frame_dataset"
            result["mAP50_note"] = None
        except Exception as error:
            result["mAP50"] = None
            result["mAP50_note"] = f"Không chạy được validation: {error}"
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_density_levels(
    prediction_path: str | Path,
    ground_truth_path: str | Path,
    output_path: str | Path,
    tolerance_seconds: float = 0.6,
) -> dict:
    output_file = Path(output_path)
    result = json.loads(output_file.read_text(encoding="utf-8")) if output_file.is_file() else {}
    truth_path = Path(ground_truth_path)
    if not truth_path.is_file():
        result["density_macro_f1"] = None
        result["density_confusion_matrix"] = None
        result["density_note"] = "Chưa có annotations/density.csv"
    else:
        predictions = pd.read_csv(prediction_path)
        truth = pd.read_csv(truth_path)
        required = {"timestamp_s", "density_level"}
        if not required.issubset(truth.columns):
            result["density_macro_f1"] = None
            result["density_note"] = "density.csv cần timestamp_s và density_level"
        else:
            labels = ["Thấp", "Trung bình", "Cao"]
            matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
            prediction_times = predictions["timestamp_s"].to_numpy()
            for _, row in truth.iterrows():
                if row["density_level"] not in labels or not prediction_times.size:
                    continue
                index = int(np.argmin(np.abs(prediction_times - float(row["timestamp_s"]))))
                if abs(prediction_times[index] - float(row["timestamp_s"])) <= tolerance_seconds:
                    predicted = predictions.iloc[index]["density_level"]
                    if predicted in labels:
                        matrix[row["density_level"]][predicted] += 1
            f1_scores = []
            for label in labels:
                true_positive = matrix[label][label]
                false_positive = sum(matrix[other][label] for other in labels if other != label)
                false_negative = sum(matrix[label][other] for other in labels if other != label)
                denominator = 2 * true_positive + false_positive + false_negative
                f1_scores.append(2 * true_positive / denominator if denominator else 0.0)
            matched = sum(sum(row.values()) for row in matrix.values())
            result["density_macro_f1"] = float(np.mean(f1_scores)) if matched else None
            result["density_confusion_matrix"] = matrix
            result["density_note"] = None if matched else "Không ghép được timestamp"
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_roi_iou(
    predicted_roi_path: str | Path,
    reference_roi_path: str | Path,
    width: int,
    height: int,
    output_path: str | Path,
) -> dict:
    output_file = Path(output_path)
    result = json.loads(output_file.read_text(encoding="utf-8")) if output_file.is_file() else {}
    reference = load_roi(reference_roi_path, width, height)
    predicted = load_roi(predicted_roi_path, width, height)
    if reference is None or predicted is None:
        result["roi_iou"] = None
        result["roi_iou_note"] = "Chưa có annotations/roi_reference.json"
    else:
        predicted_mask = roi_mask(width, height, predicted) > 0
        reference_mask = roi_mask(width, height, reference) > 0
        union = np.logical_or(predicted_mask, reference_mask).sum()
        intersection = np.logical_and(predicted_mask, reference_mask).sum()
        result["roi_iou"] = float(intersection / union) if union else None
        result["roi_iou_note"] = None
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

