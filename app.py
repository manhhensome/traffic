from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis import AnalysisConfig, list_runs, recover_incomplete_runs
from src.dataset import create_grouped_splits, load_splits, save_split
from src.jobs import cancel_job, find_active_job, latest_job, start_analysis_job
from src.preprocessing import (
    adaptive_canny,
    apply_clahe,
    fixed_canny,
    sobel_edges,
    suggest_road_roi,
)
from src.reporting import build_experiment_reports
from src.roi import load_roi, overlay_roi, save_roi
from src.storage import ALLOWED_EXTENSIONS, VideoStorage
from src.video_io import extract_frames, inspect_video, read_frame


ROOT_DIRECTORY = Path(__file__).parent

st.set_page_config(
    page_title="Phân tích mật độ giao thông Hà Nội",
    page_icon="🚦",
    layout="wide",
)


@st.cache_resource
def get_storage() -> VideoStorage:
    storage_instance = VideoStorage(ROOT_DIRECTORY / "data")
    recover_incomplete_runs(ROOT_DIRECTORY / "data")
    return storage_instance


@st.cache_data(show_spinner=False)
def get_video_info(path: str, modified_at: float):
    del modified_at
    return inspect_video(path)


storage = get_storage()

st.title("Đánh giá mật độ giao thông qua video")
st.caption("Tách biên thích nghi, ROI, MOG2 và YOLOv8 cho video giao thông đô thị.")

with st.sidebar:
    st.header("1. Chọn dữ liệu")
    uploaded_files = st.file_uploader(
        "Chọn video từ máy tính",
        type=sorted(extension.removeprefix(".") for extension in ALLOWED_EXTENSIONS),
        accept_multiple_files=True,
        help="Mỗi video mới được lưu vào data/video-N.",
    )
    if st.button(
        "Lưu video đã chọn",
        type="primary",
        disabled=not uploaded_files,
        width="stretch",
    ):
        progress = st.progress(0, text="Đang lưu video...")
        saved_count = duplicate_count = 0
        for index, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                video, created = storage.save_upload(uploaded_file, uploaded_file.name)
                inspect_video(video.file_path)
                saved_count += int(created)
                duplicate_count += int(not created)
                st.session_state["selected_video_id"] = video.video_id
            except (OSError, ValueError) as error:
                st.error(f"{uploaded_file.name}: {error}")
            progress.progress(index / len(uploaded_files), text=f"Đã xử lý {index}/{len(uploaded_files)}")
        progress.empty()
        if saved_count:
            st.success(f"Đã lưu {saved_count} video mới.")
        if duplicate_count:
            st.info(f"{duplicate_count} video đã tồn tại nên không lưu trùng.")

    videos = storage.list_videos()
    if not videos:
        st.info("Chưa có video. Hãy upload một video để bắt đầu.")
        st.stop()

    video_by_id = {video.video_id: video for video in videos}
    default_id = st.session_state.get("selected_video_id", videos[0].video_id)
    if default_id not in video_by_id:
        default_id = videos[0].video_id
    selected_id = st.selectbox(
        "Video đang làm việc",
        options=list(video_by_id),
        index=list(video_by_id).index(default_id),
        format_func=lambda video_id: video_by_id[video_id].label,
    )
    st.session_state["selected_video_id"] = selected_id
    with st.expander("Chia tập dữ liệu"):
        current_splits = load_splits(ROOT_DIRECTORY / "data")
        current_split = current_splits.get(str(selected_id), "train")
        split = st.selectbox(
            "Split của video",
            ["train", "val", "test"],
            index=["train", "val", "test"].index(current_split),
            key=f"split-{selected_id}",
        )
        if st.button("Lưu split", key=f"save-split-{selected_id}", width="stretch"):
            save_split(ROOT_DIRECTORY / "data", selected_id, split)
            st.success("Đã lưu splits.json")
        if st.button("Tạo chia 60/20/20 theo phiên quay", width="stretch"):
            metadata_items = [storage.read_metadata(video.video_id) for video in videos]
            create_grouped_splits(ROOT_DIRECTORY / "data", metadata_items)
            st.success("Đã tạo splits.json theo recording_session_id.")

selected_video = video_by_id[selected_id]
try:
    video_info = get_video_info(
        str(selected_video.file_path), selected_video.file_path.stat().st_mtime
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

video_tab, roi_tab, analysis_tab, result_tab = st.tabs(
    ["Video", "ROI & tách biên", "Phân tích", "Kết quả"]
)

with video_tab:
    video_column, info_column = st.columns([2, 1])
    with video_column:
        st.video(str(selected_video.file_path))
    with info_column:
        st.subheader(f"video-{selected_video.video_id}")
        st.write(f"**Tên gốc:** {selected_video.original_filename}")
        st.write(f"**Kích thước:** {video_info.width} × {video_info.height}")
        st.write(f"**FPS:** {video_info.fps:.2f}")
        st.write(f"**Thời lượng:** {video_info.duration_seconds:.1f} giây")
        st.write(f"**Số frame:** {video_info.frame_count:,}")
        st.write(f"**Dung lượng:** {selected_video.size_bytes / (1024 * 1024):.2f} MB")
        if not 300 <= video_info.duration_seconds <= 600:
            st.warning("Clip nằm ngoài khoảng 5–10 phút đề xuất, nhưng vẫn có thể phân tích.")
        extraction_interval = st.number_input(
            "Khoảng trích frame (giây)", min_value=0.2, value=1.0, step=0.2
        )
        if st.button("Trích frame để gán nhãn", width="stretch"):
            with st.spinner("Đang trích frame..."):
                manifest = extract_frames(
                    selected_video.file_path,
                    selected_video.directory / "frames",
                    extraction_interval,
                )
            frame_count = len(pd.read_csv(manifest))
            st.success(f"Đã trích {frame_count} frame vào data/video-{selected_id}/frames")

    metadata = storage.read_metadata(selected_id)
    st.subheader("Bối cảnh ghi hình")
    with st.form(f"metadata-form-{selected_id}"):
        metadata_a, metadata_b, metadata_c = st.columns(3)
        with metadata_a:
            intersection = st.text_input("Nút giao", value=metadata.get("intersection") or "")
            camera_id = st.text_input("Camera ID", value=metadata.get("camera_id") or "")
            recording_session_id = st.text_input(
                "Mã phiên quay", value=metadata.get("recording_session_id") or ""
            )
        with metadata_b:
            lighting_options = ["unknown", "day", "night"]
            lighting = st.selectbox(
                "Điều kiện sáng",
                lighting_options,
                index=lighting_options.index(metadata.get("lighting", "unknown"))
                if metadata.get("lighting", "unknown") in lighting_options else 0,
            )
            weather_options = ["unknown", "clear", "rain", "fog"]
            weather = st.selectbox(
                "Thời tiết",
                weather_options,
                index=weather_options.index(metadata.get("weather", "unknown"))
                if metadata.get("weather", "unknown") in weather_options else 0,
            )
            recorded_at = st.text_input(
                "Thời điểm quay (ISO 8601)", value=metadata.get("recorded_at") or ""
            )
        with metadata_c:
            source_options = ["upload", "phone", "camera", "public_livestream"]
            source_type = st.selectbox(
                "Nguồn video",
                source_options,
                index=source_options.index(metadata.get("source_type", "upload"))
                if metadata.get("source_type", "upload") in source_options else 0,
            )
            source_url = st.text_input("URL nguồn công khai", value=metadata.get("source_url") or "")
        if st.form_submit_button("Lưu metadata", type="primary"):
            storage.update_metadata(
                selected_id,
                {
                    "intersection": intersection or None,
                    "camera_id": camera_id or None,
                    "recording_session_id": recording_session_id or None,
                    "lighting": lighting,
                    "weather": weather,
                    "recorded_at": recorded_at or None,
                    "source_type": source_type,
                    "source_url": source_url or None,
                },
            )
            st.success("Đã cập nhật metadata.")

with roi_tab:
    st.subheader("Chọn vùng mặt đường")
    preview_time = st.slider(
        "Thời điểm lấy frame xem trước (giây)",
        0.0,
        max(0.1, float(video_info.duration_seconds)),
        min(float(video_info.duration_seconds / 3), float(video_info.duration_seconds)),
        step=max(0.1, min(1.0, video_info.duration_seconds / 100 if video_info.duration_seconds else 0.1)),
        key=f"preview-time-{selected_id}",
    )
    preview_frame = read_frame(selected_video.file_path, preview_time)
    roi_path = selected_video.directory / "roi.json"
    default_points = load_roi(roi_path, video_info.width, video_info.height)
    if default_points is None:
        default_points = suggest_road_roi(preview_frame)

    point_keys = [(f"roi-{selected_id}-x-{index}", f"roi-{selected_id}-y-{index}") for index in range(4)]
    for (x_key, y_key), (default_x, default_y) in zip(point_keys, default_points):
        if x_key not in st.session_state:
            st.session_state[x_key] = default_x
        if y_key not in st.session_state:
            st.session_state[y_key] = default_y

    control_a, control_b = st.columns([1, 3])
    with control_a:
        use_clahe_preview = st.checkbox("Áp dụng CLAHE", value=False, key=f"clahe-preview-{selected_id}")
        edge_method = st.radio(
            "Tách biên",
            ["Canny thích nghi", "Canny cố định", "Sobel"],
            key=f"edge-{selected_id}",
        )
        fixed_low = st.slider("Canny thấp", 0, 254, 100, disabled=edge_method != "Canny cố định")
        fixed_high = st.slider("Canny cao", 1, 255, 200, disabled=edge_method != "Canny cố định")
        if st.button("Gợi ý lại ROI", key=f"suggest-{selected_id}", width="stretch"):
            sample_times = [
                video_info.duration_seconds * ratio for ratio in (0.2, 0.5, 0.8)
            ] if video_info.duration_seconds > 0 else [0.0]
            candidate_rois = []
            for sample_time in sample_times:
                sample_frame = read_frame(selected_video.file_path, sample_time)
                if use_clahe_preview:
                    sample_frame = apply_clahe(sample_frame)
                candidate_rois.append(
                    suggest_road_roi(
                        sample_frame,
                        edge_mode="fixed" if edge_method == "Canny cố định" else "adaptive",
                        fixed_thresholds=(min(fixed_low, fixed_high - 1), fixed_high),
                    )
                )
            suggestions = [
                (int(point[0]), int(point[1]))
                for point in np.median(np.asarray(candidate_rois), axis=0)
            ]
            save_roi(
                selected_video.directory / "roi_suggestion.json",
                suggestions,
                video_info.width,
                video_info.height,
            )
            (selected_video.directory / "roi_suggestion_meta.json").write_text(
                json.dumps(
                    {
                        "method": edge_method,
                        "use_clahe": use_clahe_preview,
                        "preview_timestamp_s": preview_time,
                        "fixed_thresholds": [fixed_low, fixed_high],
                        "sample_timestamps_s": sample_times,
                        "suggested_at_unix": time.time(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            for (x_key, y_key), (x_value, y_value) in zip(point_keys, suggestions):
                st.session_state[x_key] = x_value
                st.session_state[y_key] = y_value
            st.rerun()

        st.markdown("**Bốn đỉnh ROI**")
        for index, (x_key, y_key) in enumerate(point_keys, start=1):
            x_column, y_column = st.columns(2)
            x_column.number_input(f"X{index}", 0, video_info.width - 1, key=x_key)
            y_column.number_input(f"Y{index}", 0, video_info.height - 1, key=y_key)

    roi_points = [
        (int(st.session_state[x_key]), int(st.session_state[y_key])) for x_key, y_key in point_keys
    ]
    processed_preview = apply_clahe(preview_frame) if use_clahe_preview else preview_frame
    with control_b:
        image_a, image_b = st.columns(2)
        try:
            image_a.image(
                overlay_roi(preview_frame, roi_points),
                channels="BGR",
                caption="ROI trên frame gốc",
                width="stretch",
            )
        except ValueError as error:
            image_a.error(str(error))
        if edge_method == "Canny thích nghi":
            edges, low_threshold, high_threshold = adaptive_canny(processed_preview)
            caption = f"Canny: ngưỡng {low_threshold}/{high_threshold}"
        elif edge_method == "Canny cố định":
            low_threshold = min(fixed_low, fixed_high - 1)
            edges = fixed_canny(processed_preview, low_threshold, fixed_high)
            caption = f"Canny cố định: {low_threshold}/{fixed_high}"
        else:
            edges = sobel_edges(processed_preview)
            caption = "Biên Sobel"
        image_b.image(edges, caption=caption, width="stretch")

    if st.button("Lưu ROI", type="primary", key=f"save-roi-{selected_id}"):
        try:
            save_roi(roi_path, roi_points, video_info.width, video_info.height)
            suggestion_meta_path = selected_video.directory / "roi_suggestion_meta.json"
            if suggestion_meta_path.is_file():
                suggestion_meta = json.loads(suggestion_meta_path.read_text(encoding="utf-8"))
                suggestion_meta["manual_edit_seconds"] = max(
                    0.0, time.time() - float(suggestion_meta.get("suggested_at_unix", time.time()))
                )
                suggestion_meta["saved_roi_at_unix"] = time.time()
                suggestion_meta_path.write_text(
                    json.dumps(suggestion_meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            st.success(f"Đã lưu ROI tại data/video-{selected_id}/roi.json")
        except ValueError as error:
            st.error(str(error))

with analysis_tab:
    st.subheader("Cấu hình và chạy phân tích")
    saved_roi = load_roi(selected_video.directory / "roi.json", video_info.width, video_info.height)
    if saved_roi is None:
        st.warning("Hãy sang tab **ROI & tách biên** và lưu ROI trước khi phân tích.")
    method_label = st.selectbox(
        "Phương pháp",
        ["YOLOv8 — nhận dạng phương tiện", "MOG2 — baseline chuyển động"],
    )
    method = "yolo" if method_label.startswith("YOLO") else "mog2"
    config_left, config_right = st.columns(2)
    with config_left:
        frame_stride = st.slider("Phân tích mỗi N frame", 1, 30, 5)
        max_seconds = st.number_input(
            "Giới hạn thời lượng (giây, 0 = toàn bộ)", min_value=0.0, value=30.0, step=10.0
        )
        use_clahe = st.checkbox("CLAHE trước khi nhận dạng", value=False)
        hysteresis_margin = st.slider("Biên hysteresis mật độ", 0.00, 0.10, 0.02, 0.01)
    with config_right:
        confidence = st.slider("Ngưỡng confidence YOLO", 0.05, 0.95, 0.35, 0.05, disabled=method != "yolo")
        image_size = st.select_slider("Kích thước ảnh YOLO", [416, 512, 640, 768, 960], value=640, disabled=method != "yolo")
        device = st.selectbox("Thiết bị", ["cpu", "0 (CUDA)"], disabled=method != "yolo")
        mog2_warmup = st.number_input(
            "Khởi tạo nền MOG2 (giây)", 0.0, 30.0, 2.0, 0.5, disabled=method != "mog2"
        )
    threshold_left, threshold_right = st.columns(2)
    medium_threshold = threshold_left.slider("Ngưỡng mật độ trung bình", 0.01, 0.50, 0.12, 0.01)
    high_threshold = threshold_right.slider("Ngưỡng mật độ cao", 0.02, 0.80, 0.28, 0.01)
    if high_threshold <= medium_threshold:
        st.error("Ngưỡng mật độ cao phải lớn hơn ngưỡng trung bình.")

    can_run = saved_roi is not None and high_threshold > medium_threshold
    current_job = latest_job(selected_video.directory)
    if current_job is not None:
        current_job_path, current_job_payload = current_job
        status = current_job_payload.get("status", "unknown")
        st.write(f"**Tác vụ gần nhất:** `{status}` — {current_job_payload.get('message', '')}")
        st.progress(float(current_job_payload.get("progress", 0.0)))
        job_a, job_b = st.columns(2)
        if job_a.button("Làm mới trạng thái", width="stretch"):
            st.rerun()
        if job_b.button(
            "Hủy tác vụ",
            disabled=status not in {"queued", "processing"},
            width="stretch",
        ):
            cancel_job(current_job_path)
            st.rerun()

    global_active_job = find_active_job(ROOT_DIRECTORY / "data")
    if st.button(
        "Chạy phân tích",
        type="primary",
        disabled=not can_run or global_active_job is not None,
        width="stretch",
    ):
        config = AnalysisConfig(
            method=method,
            device="0" if device.startswith("0") else "cpu",
            confidence=confidence,
            image_size=image_size,
            frame_stride=frame_stride,
            max_seconds=max_seconds,
            use_clahe=use_clahe,
            medium_threshold=medium_threshold,
            high_threshold=high_threshold,
            hysteresis_margin=hysteresis_margin,
            mog2_warmup_seconds=mog2_warmup,
        )
        try:
            job_path = start_analysis_job(
                ROOT_DIRECTORY,
                selected_video.file_path,
                selected_video.directory,
                saved_roi,
                config,
            )
            st.success(f"Đã khởi động {job_path.stem}. Có thể làm mới trạng thái hoặc chuyển tab.")
            st.rerun()
        except Exception as error:
            st.error(f"Không khởi động được tác vụ: {error}")

with result_tab:
    st.subheader("Nhãn tham chiếu")
    label_a, label_b, label_c = st.columns(3)
    with label_a:
        count_labels = st.file_uploader(
            "counts.csv — MAE",
            type=["csv"],
            key=f"count-labels-{selected_id}",
            help="Cần timestamp_s, n_total; có thể thêm các lớp phương tiện.",
        )
    with label_b:
        density_labels = st.file_uploader(
            "density.csv — macro-F1",
            type=["csv"],
            key=f"density-labels-{selected_id}",
            help="Cần timestamp_s và density_level: Thấp/Trung bình/Cao.",
        )
    with label_c:
        roi_reference = st.file_uploader(
            "roi_reference.json — ROI IoU",
            type=["json"],
            key=f"roi-reference-{selected_id}",
        )
    if st.button(
        "Lưu các nhãn đã chọn",
        disabled=not any([count_labels, density_labels, roi_reference]),
        key=f"save-labels-{selected_id}",
    ):
        annotations_directory = selected_video.directory / "annotations"
        annotations_directory.mkdir(exist_ok=True)
        try:
            if count_labels is not None:
                labels_table = pd.read_csv(count_labels)
                if not {"timestamp_s", "n_total"}.issubset(labels_table.columns):
                    raise ValueError("counts.csv cần timestamp_s và n_total.")
                labels_table.to_csv(annotations_directory / "counts.csv", index=False)
            if density_labels is not None:
                density_table = pd.read_csv(density_labels)
                if not {"timestamp_s", "density_level"}.issubset(density_table.columns):
                    raise ValueError("density.csv cần timestamp_s và density_level.")
                density_table.to_csv(annotations_directory / "density.csv", index=False)
            if roi_reference is not None:
                payload = json.load(roi_reference)
                if "points" not in payload:
                    raise ValueError("roi_reference.json cần trường points.")
                (annotations_directory / "roi_reference.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            st.success("Đã lưu nhãn. Chạy lại phân tích để cập nhật đánh giá.")
        except Exception as error:
            st.error(f"Không lưu được nhãn: {error}")

    runs = list_runs(selected_video.directory)
    if not runs:
        st.info("Chưa có lần phân tích hoàn tất cho video này.")
    else:
        run_by_name = {run.name: run for run in runs}
        selected_run_name = st.selectbox(
            "Lần chạy", list(run_by_name), key=f"selected-run-{selected_id}"
        )
        selected_run = run_by_name[selected_run_name]
        summary = json.loads((selected_run / "summary.json").read_text(encoding="utf-8"))
        evaluation = json.loads((selected_run / "evaluation.json").read_text(encoding="utf-8"))
        table = pd.read_csv(selected_run / "timeseries.csv")

        metric_columns = st.columns(4)
        metric_columns[0].metric("Trung bình xe/vật thể", f"{summary['mean_vehicle_or_object_count']:.2f}")
        metric_columns[1].metric("Chiếm dụng trung bình", f"{summary['mean_occupancy']:.1%}")
        metric_columns[2].metric("FPS xử lý", f"{summary['processing_fps']:.2f}")
        metric_columns[3].metric("Frame đã lấy mẫu", f"{summary['sampled_frames']:,}")

        chart_column, video_result_column = st.columns([1, 1])
        with chart_column:
            st.markdown("**Số xe/vật thể theo thời gian**")
            count_columns = [
                column for column in ["n_motorcycle", "n_car", "n_bus", "n_truck", "n_total"]
                if column in table.columns
            ]
            st.line_chart(table.set_index("timestamp_s")[count_columns])
            st.markdown("**Tỷ lệ chiếm dụng ROI**")
            st.line_chart(table.set_index("timestamp_s")[["occupancy", "occupancy_smooth"]])
        with video_result_column:
            st.video(str(selected_run / "annotated.mp4"))
            st.write("**Phân bố mật độ:**", summary["density_distribution"])
            if evaluation["status"] == "evaluated":
                st.write("**MAE đếm xe:**", evaluation["count_mae"])
            else:
                st.caption("Chưa đánh giá MAE: thêm annotations/counts.csv cho video.")
            if evaluation.get("mAP50") is not None:
                st.metric("mAP@0.5", f"{evaluation['mAP50']:.3f}")
            else:
                st.caption(evaluation.get("mAP50_note", "Chưa có nhãn bounding box để tính mAP."))
            if evaluation.get("density_macro_f1") is not None:
                st.metric("Macro-F1 mật độ", f"{evaluation['density_macro_f1']:.3f}")
            if evaluation.get("roi_iou") is not None:
                st.metric("ROI IoU", f"{evaluation['roi_iou']:.3f}")

        download_a, download_b, download_c = st.columns(3)
        download_a.download_button(
            "Tải timeseries.csv",
            (selected_run / "timeseries.csv").read_bytes(),
            file_name=f"{selected_run_name}-timeseries.csv",
            mime="text/csv",
            width="stretch",
        )
        download_b.download_button(
            "Tải detections.csv",
            (selected_run / "detections.csv").read_bytes(),
            file_name=f"{selected_run_name}-detections.csv",
            mime="text/csv",
            width="stretch",
        )
        download_c.download_button(
            "Tải summary.json",
            (selected_run / "summary.json").read_bytes(),
            file_name=f"{selected_run_name}-summary.json",
            mime="application/json",
            width="stretch",
        )

    st.divider()
    st.subheader("So sánh thí nghiệm")
    if st.button("Tạo báo cáo tổng hợp", width="stretch"):
        detail_report, condition_report = build_experiment_reports(
            ROOT_DIRECTORY / "data", ROOT_DIRECTORY / "reports"
        )
        st.session_state["detail-report"] = str(detail_report)
        st.session_state["condition-report"] = str(condition_report)
    detail_report_path = Path(st.session_state.get("detail-report", ROOT_DIRECTORY / "reports" / "experiment_runs.csv"))
    condition_report_path = Path(st.session_state.get("condition-report", ROOT_DIRECTORY / "reports" / "condition_summary.csv"))
    if detail_report_path.is_file():
        detail_report_table = pd.read_csv(detail_report_path)
        if detail_report_table.empty:
            st.info("Chưa có run hoàn tất để tổng hợp.")
        else:
            st.dataframe(detail_report_table, width="stretch", hide_index=True)
            if condition_report_path.is_file() and condition_report_path.stat().st_size:
                condition_table = pd.read_csv(condition_report_path)
                st.markdown("**Trung bình theo ngày/đêm, phương pháp và CLAHE**")
                st.dataframe(condition_table, width="stretch", hide_index=True)
            st.download_button(
                "Tải báo cáo thí nghiệm",
                detail_report_path.read_bytes(),
                file_name="experiment_runs.csv",
                mime="text/csv",
            )
