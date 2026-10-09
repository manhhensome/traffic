# Đối chiếu triển khai với PLAN.md

Ngày kiểm tra: 01/10/2026.

## Đã hoàn thành trong mã nguồn

- Upload nhiều video, kiểm tra khả năng giải mã, SHA-256 chống trùng, SQLite cấp
  ID tăng dần và lưu `data/video-N`.
- Metadata đầy đủ cho nguồn, nút giao, camera, phiên quay, ngày/đêm, thời tiết,
  thời điểm quay và thông số kỹ thuật video; có thumbnail và cảnh báo clip ngoài
  5–10 phút.
- Trích frame theo khoảng thời gian và tạo `frames/manifest.csv`.
- ROI bốn đỉnh, tọa độ chuẩn hóa, kiểm tra diện tích/biên/tự cắt, xem trước và
  lưu bản đóng băng cho từng run.
- CLAHE, Sobel, Canny cố định và Canny thích nghi theo phân vị gradient; gợi ý
  ROI từ Hough trên ba thời điểm của video. Lưu ROI trước chỉnh tay, phương pháp
  biên và thời gian chỉnh sửa.
- Baseline MOG2 có loại bóng, morphology, warm-up, số cụm và chiếm dụng theo
  foreground mask.
- YOLOv8n lọc bốn lớp `motorcycle`, `car`, `bus`, `truck` bằng điểm giữa cạnh đáy
  trong ROI; trọng số đặt tại `models/yolov8n.pt`.
- Đếm/chiếm dụng từng frame mẫu, tổng hợp thành cửa sổ một giây, trung bình trượt
  năm giây và hysteresis cho mức Thấp/Trung bình/Cao.
- Video phủ kết quả, `detections.csv`, `frame_metrics.csv`, `timeseries.csv`,
  `summary.json`, `evaluation.json`, mask ROI và cấu hình run.
- Trạng thái `queued/processing/completed/failed/interrupted`, `run.log`, nhận
  diện run dở khi ứng dụng khởi động lại.
- Worker nền chỉ cho phép một tác vụ hoạt động, lưu tiến độ, hỗ trợ làm mới và
  hủy từ giao diện; video không phải upload lại khi đổi cấu hình.
- Lưu phiên bản Python/thư viện, phần cứng, class mapping và SHA-256 trọng số.
- MAE từ `counts.csv`; confusion matrix và macro-F1 từ `density.csv`; ROI IoU từ
  `roi_reference.json`; mAP@0.5/mAP@0.5:0.95 từ dataset YOLO.
- FPS lấy mẫu, FPS toàn pipeline, latency trung vị/p95, độ lệch chuẩn và hệ số
  biến thiên FPS.
- Chia train/val/test 60/20/20 theo `recording_session_id`, không tách frame của
  cùng phiên quay sang nhiều tập.
- Báo cáo so sánh MOG2/YOLO, CLAHE và ngày/đêm trong `reports/`.
- CLI fine-tune YOLOv8 với seed cố định qua `train.py`.

## Chỉ hoàn tất khi có dữ liệu thật

Các đầu việc sau không thể tạo kết quả hợp lệ chỉ bằng mã nguồn:

- Thu thập 12–18 video tại ít nhất ba nút giao Hà Nội, đủ ngày/đêm và điều kiện
  giao thông; hiện thư mục dữ liệu chưa có video thật do người dùng cung cấp.
- Gán 600–1.000 frame, đếm xe, nhãn mật độ và ROI tham chiếu; kiểm tra chéo
  10–20% giữa hai người.
- Chạy bảng thí nghiệm A–F trên train/val/test, hiệu chỉnh confidence và ngưỡng
  mật độ trên validation rồi khóa test.
- Báo cáo các giá trị mAP, MAE, macro-F1, ROI IoU và FPS ngày/đêm thực tế. Ứng
  dụng giữ các giá trị này là `null` khi chưa có ground truth.
- Quyết định fine-tune dựa trên lỗi xe máy của mô hình pretrained. `train.py` đã
  sẵn sàng nhưng không tự huấn luyện khi chưa có nhãn.

## Phần mở rộng ngoài MVP

ByteTrack/vạch đếm lưu lượng, YOLOv8-seg, hiệu chuẩn mật độ vật lý và mô phỏng
chu kỳ đèn được PLAN.md ghi là phần mở rộng. Chúng chưa được bật trong giao diện
MVP và không ảnh hưởng các đầu ra bắt buộc về số xe hiện diện, chiếm dụng và mức
mật độ.
