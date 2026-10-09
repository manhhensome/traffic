# Kế hoạch đồ án đánh giá mật độ giao thông tại nút giao Hà Nội

Ngày lập: 01/10/2026. Thời gian: 5 tuần kể từ ngày bắt đầu thực hiện.

Đây là kế hoạch triển khai, chưa phải ứng dụng đã chạy. Dự án hiện có README trống và chưa có mã nguồn xử lý video. Thiết kế dưới đây đáp ứng yêu cầu: mỗi video upload được lưu trong một thư mục riêng `data/video-1`, `data/video-2`, …; dữ liệu và kết quả của video được quản lý cùng nhau.

## 1. Mục tiêu và phạm vi

Xây dựng ứng dụng chạy cục bộ, cho phép upload video giao thông Hà Nội dài khoảng 5–10 phút, chọn vùng mặt đường, nhận dạng phương tiện và xem biến động mật độ theo thời gian.

Đầu ra bắt buộc của MVP:

- Video gốc và metadata được lưu bền vững, có thể mở lại sau khi khởi động ứng dụng.
- ROI đa giác cho toàn vùng mặt đường hoặc từng hướng tiếp cận nút giao.
- Số xe máy, ô tô, xe buýt, xe tải đang hiện diện trong từng ROI.
- Tỷ lệ chiếm dụng ảnh trong ROI; mức mật độ thấp/vừa/cao với ngưỡng đã hiệu chỉnh.
- Video có lớp phủ nhận dạng, biểu đồ theo thời gian, CSV và báo cáo JSON.
- So sánh baseline MOG2 với YOLOv8 kết hợp ROI và tiền xử lý.
- Đánh giá mAP@0.5, MAE đếm xe và FPS riêng cho ngày/đêm.

Phần mở rộng nếu còn thời gian: ByteTrack và vạch đếm để đo lưu lượng xe đi qua; YOLOv8-seg để ước lượng diện tích tốt hơn; mô phỏng phân bổ thời gian đèn theo từng hướng. Điều khiển đèn thực tế nằm ngoài phạm vi đồ án 5 tuần.

## 2. Phân biệt các đại lượng cần đo

| Đại lượng | Cách tính | Ý nghĩa và giới hạn |
| --- | --- | --- |
| Số xe hiện diện `N_c(t)` | Đếm detection thuộc lớp `c`, có điểm giữa cạnh đáy nằm trong ROI tại thời điểm `t` | Một xe có thể xuất hiện ở nhiều frame; không cộng các frame để tính tổng lượt xe |
| Chiếm dụng ảnh `O(t)` | Diện tích hợp các bounding box, cắt theo ROI, chia diện tích ROI | Chỉ số thay thế cho mức chiếm dụng; box chứa cả nền và chịu ảnh hưởng phối cảnh |
| Lưu lượng, tùy chọn | Số track vượt vạch theo chiều quy định trong một phút | Cần tracking, kiểm tra đổi phía và chống đếm lặp; không dùng số ID xuất hiện làm lưu lượng |
| Mật độ vật lý, mở rộng | Số xe / diện tích mặt đường đã hiệu chuẩn | Chỉ báo cáo xe/m² hoặc xe/km khi đã có hiệu chuẩn hình học phù hợp |

Công thức chiếm dụng:

```text
R       = mặt nạ ROI cố định trong một lần phân tích
B_i(t)  = bounding box của phương tiện i được chấp nhận trong ROI
O(t)    = area(R ∩ union_i B_i(t)) / area(R)
```

Vẽ các box vào cùng một mặt nạ nhị phân để phần giao nhau chỉ được tính một lần. `O(t)` phải nằm trong `[0, 1]`. ROI rỗng là lỗi cấu hình; frame không đọc được là dữ liệu thiếu, không phải `N=0`.

Xuất số đo theo cửa sổ 1 giây thời gian video; dùng trung bình trượt khoảng 5 giây và hysteresis để mức mật độ ít nhấp nháy. Các thời lượng là tham số khởi đầu, cần hiệu chỉnh trên validation. Không so sánh trực tiếp chiếm dụng giữa các góc camera khác nhau nếu chưa chuẩn hóa.

Chỉ gọi đầu ra cơ bản là **mức mật độ**. Muốn kết luận **nghi tắc nghẽn**, cần bổ sung chuyển động thấp kéo dài, nhãn tham chiếu và bối cảnh chu kỳ đèn; hàng xe dừng khi đèn đỏ không tự động đồng nghĩa với tắc nghẽn. Không đặt ngưỡng phần trăm chung cho mọi camera.

## 3. Luồng sử dụng và lưu trữ video

```text
Upload → Kiểm tra video → Cấp video_id và lưu gốc
       → Xem trước, khai báo bối cảnh → Vẽ/duyệt ROI
       → Chọn cấu hình → Phân tích → Xem và tải kết quả
```

Một file upload tương ứng một thư mục. Nếu chọn nhiều file, xử lý thành nhiều hồ sơ video độc lập. Video gốc giữ nguyên nội dung; không đổi đuôi MOV thành MP4 nếu chưa chuyển mã.

```text
traffic/
├── app.py                       # Giao diện dự kiến
├── src/
│   ├── storage.py               # Upload, ID, metadata, trạng thái
│   ├── video_io.py              # Đọc video, timestamp, frame mẫu
│   ├── preprocessing.py         # CLAHE, Gaussian, Sobel/Canny
│   ├── roi.py                   # Đề xuất/vẽ ROI, kiểm tra hình học
│   ├── baseline.py              # MOG2 và vùng chuyển động
│   ├── detector.py              # YOLOv8, lọc lớp phương tiện
│   ├── density.py               # Đếm, chiếm dụng, tổng hợp thời gian
│   ├── evaluation.py            # mAP, MAE, ROI IoU, FPS
│   └── reporting.py             # Video phủ kết quả, CSV, JSON
├── configs/default.yaml
├── models/                      # Trọng số dùng chung
├── data/
│   ├── catalog.sqlite          # Chỉ mục video và cấp ID tăng dần
│   ├── splits.json             # Nhóm train/val/test, không sao chép video
│   ├── video-1/
│   │   ├── original.mp4        # Hoặc original.mov / original.avi
│   │   ├── metadata.json
│   │   ├── thumbnail.jpg
│   │   ├── roi.json            # ROI hiện được người dùng duyệt
│   │   ├── frames/
│   │   │   ├── manifest.csv    # Tên frame, frame index, timestamp
│   │   │   └── frame_000001.jpg
│   │   ├── annotations/
│   │   │   ├── labels/         # Box tham chiếu theo định dạng YOLO
│   │   │   ├── counts.csv      # Đếm tay theo timestamp và ROI
│   │   │   └── density.csv     # Mức mật độ do người gán nhãn
│   │   └── runs/
│   │       └── run-0001/
│   │           ├── config.yaml
│   │           ├── roi.json    # Bản chụp ROI sử dụng trong lần chạy
│   │           ├── roi_mask.png
│   │           ├── detections.csv
│   │           ├── timeseries.csv
│   │           ├── summary.json
│   │           ├── evaluation.json
│   │           ├── annotated.mp4
│   │           └── run.log
│   └── video-2/
│       └── ...
├── reports/                    # Báo cáo tổng hợp nhiều video
└── PLAN.md
```

Quy tắc upload cần triển khai:

1. MVP ưu tiên MP4/H.264; MOV/AVI được nhận nếu bộ giải mã hỗ trợ. Kiểm tra nội dung giải mã, kích thước frame, thời lượng và khả năng đọc mẫu; không chỉ kiểm tra phần mở rộng. Lỗi giữa video phải được ghi nhận khi chạy.
2. Đặt giới hạn dung lượng cấu hình được, đề xuất ban đầu 1 GB/file. Ghi file theo luồng hoặc file tạm, tránh tạo thêm nhiều bản sao trong RAM. Clip ngoài 5–10 phút được thông báo nhưng vẫn có thể dùng thử.
3. Cấp số tăng dần bằng giao dịch SQLite, bảo đảm upload đồng thời không trùng `video-N`; không dùng `len(thư_mục) + 1`. ID đã cấp không tái sử dụng; số thứ tự có thể có khoảng trống sau upload lỗi.
4. Lưu vào file tạm trong thư mục video, kiểm tra xong mới chuyển thành tên gốc chuẩn. Không dùng tên file do người dùng gửi làm đường dẫn; tên ban đầu chỉ lưu trong metadata.
5. SHA-256 giúp phát hiện upload trùng. Mặc định trả về hồ sơ video đã tồn tại khi file giống hệt; nếu cần phân tích lại thì tạo `run` mới, không ghi đè run cũ.
6. Trạng thái video: `uploading`, `ready`, `failed`; trạng thái phân tích lưu riêng theo run: `queued`, `processing`, `completed`, `failed`, `interrupted`. Khi khởi động lại, nhận biết tác vụ dở và cho chạy lại.
7. Chỉ tạo thư mục kết quả khi có dữ liệu tương ứng. Chưa có nhãn thì `evaluation.json` ghi `not_evaluated` và metric `null`; tuyệt đối không tạo mAP/MAE giả.
8. Khi triển khai, bổ sung `.gitignore` cho `data/video-*/`, catalog và trọng số. Quy tắc hiện tại trong dự án chỉ bao phủ các đường dẫn dữ liệu cũ.

Metadata tối thiểu: `schema_version`, `video_id`, `original_filename`, `stored_filename`, `sha256`, `size_bytes`, `uploaded_at`, `recorded_at` nếu biết, `source_type`, `source_url` nếu có, `intersection`, `camera_id`, `recording_session_id`, `lighting`, `weather`, `width`, `height`, `fps`, `duration_seconds`, `frame_count`, `status` và thông tin lỗi. Thời điểm ghi hình và thời điểm upload là hai trường khác nhau; dùng ISO 8601 kèm múi giờ.

`roi.json` lưu các đa giác, tên hướng giao thông, tọa độ chuẩn hóa `[0,1]`, kích thước ảnh tham chiếu, phiên bản ROI và vạch đếm nếu có. Đổi ROI tạo phiên bản mới. Mỗi run giữ bản sao ROI, tham số, class mapping, hash trọng số, phiên bản thư viện và thông tin phần cứng để tái lập.

## 4. Dữ liệu và ground truth

Mục tiêu khởi đầu: 12–18 video dài 5–10 phút tại ít nhất 3 nút giao, có nhiều phiên ghi hình độc lập; ưu tiên đủ ngày, đêm, thưa xe, đông xe và dòng xe dừng/chạy. Mưa là nhóm mở rộng nếu thu thập được; không kết luận về mưa khi chưa có dữ liệu.

Giữ camera cố định, góc nhìn đủ cao và nhìn rõ vùng mặt đường. Điện thoại nên đặt trên giá. Clip lia máy, đổi góc hoặc cắt cảnh cần tách đoạn hoặc loại khỏi MVP; ROI và MOG2 giả định cảnh quan sát ổn định. Với nguồn công khai, lưu nguồn và thời điểm trích xuất, sử dụng đoạn được phép khai thác cho mục đích học tập.

Trích frame ứng viên khoảng 1 frame/giây, sau đó chọn khoảng 600–1.000 frame đa dạng để gán nhãn trong thời lượng đồ án. Tránh gán quá nhiều frame gần như trùng nhau; bảo đảm đủ ví dụ xe máy nhỏ, che khuất, ánh đèn và xe dừng. Không cần lưu tất cả frame của toàn bộ video.

Ground truth gồm:

- Bounding box từng xe và lớp `motorcycle`, `car`, `bus`, `truck`; người lái không được gán thành một xe bổ sung.
- Quy ước thống nhất với trường hợp bị che khuất và nằm sát biên ROI; chỉ suy luận trên phương tiện có bằng chứng nhìn thấy, không gán các xe hoàn toàn bị che.
- Đếm tay cùng frame, cùng ROI, cùng quy tắc điểm giữa cạnh đáy để tính MAE.
- ROI mặt đường tham chiếu vẽ thủ công, dùng đánh giá chất lượng đề xuất từ tách biên.
- Nhãn mật độ trên một tập cửa sổ thời gian, theo tiêu chí thống nhất về độ đông và chuyển động; người gán nhãn không xem kết quả thuật toán khi đánh giá.

Kiểm tra chéo khoảng 10–20% nhãn giữa hai người. Nếu chỉ có một người, thực hiện lượt rà soát độc lập sau lần gán đầu và ghi rõ hạn chế. Người thứ hai là đề xuất nhân lực cho đồ án, không phải tác vụ được tự động giao.

Chia khoảng 60/20/20 theo **phiên ghi hình hoặc video gốc**, không chia ngẫu nhiên frame liên tiếp. Các đoạn cắt từ cùng nguồn phải ở cùng tập; SHA-256 chỉ nhận ra file giống hệt, nên vẫn cần `recording_session_id` để phát hiện quan hệ giữa các clip. Cố gắng giữ ngày/đêm trong từng tập; ghi số mẫu thực tế nếu dữ liệu quá ít. Nếu đủ dữ liệu, dành một camera/nút giao riêng để kiểm tra khả năng tổng quát hóa.

Nếu fine-tune, chỉ dùng train để học; chọn tham số và ngưỡng trên validation; khóa test trước thực nghiệm cuối. Với mô hình pretrained không fine-tune, vẫn tách tập dùng hiệu chỉnh khỏi tập kiểm thử.

## 5. Phương pháp xử lý

### 5.1. Tiền xử lý và tách biên thích nghi

OpenCV cung cấp Canny với hai ngưỡng hysteresis và hỗ trợ CLAHE để tăng tương phản cục bộ. Đó là các thành phần cơ sở; quy tắc thích nghi dưới đây là thiết kế đề xuất của đồ án, cần kiểm chứng. [Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html), [CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html).

Pipeline đề xuất:

1. Đọc frame mẫu ở nhiều thời điểm; bảo toàn tỷ lệ ảnh và ghi phép biến đổi tọa độ nếu resize.
2. Tạo ảnh xám cho nhánh tách biên. Với nhánh nhận dạng, thử CLAHE trên kênh độ sáng LAB rồi chuyển lại ảnh màu; đối chứng với ảnh gốc. Khởi đầu `clipLimit=2.0`, lưới `8×8`, chỉ giữ cấu hình nếu validation có lợi.
3. Làm trơn Gaussian, tính gradient Sobel. Chọn ngưỡng cao Canny theo phân vị gradient trong vùng đường thô, ngưỡng thấp theo tỷ lệ của ngưỡng cao. Ví dụ khởi đầu `T_high=P85`, `T_low=0.4*T_high`; thống nhất thang gradient, giới hạn giá trị suy biến và hiệu chỉnh trên validation.
4. Tổng hợp bằng chứng biên từ nhiều frame; kết hợp morphology/Hough để gợi ý đường biên hoặc đa giác. Nếu dùng nền trung vị thì ghi nhận xe dừng lâu có thể vẫn xuất hiện trong nền.
5. Hiển thị ảnh gốc, biên và đa giác đề xuất để người dùng chỉnh sửa. Khi biên yếu hoặc xe che gần hết đường, dùng ROI thủ công.
6. Đóng băng ROI đã duyệt cho cả run; không tự thay đổi diện tích ROI theo từng frame vì sẽ gây biến động giả trong mẫu số chiếm dụng.

Không coi Canny/Sobel là mô hình phân đoạn mặt đường. Biên xe, bóng đổ, biển hiệu cũng tạo cạnh; kết quả tự động chỉ là đề xuất cần kiểm tra. Mật độ biên không được dùng trực tiếp như số xe.

### 5.2. Baseline MOG2

Dùng `cv2.createBackgroundSubtractorMOG2`, xử lý frame theo thứ tự, bỏ vùng ngoài ROI, loại bóng theo giá trị mask, mở/đóng hình thái rồi tìm thành phần liên thông. OpenCV mô tả trừ nền như cách tạo mặt nạ chuyển động từ camera tĩnh. [Hướng dẫn trừ nền](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html).

Lưu diện tích foreground và số cụm chuyển động. Nếu dùng số cụm làm xấp xỉ số xe, phải báo rõ đây là baseline yếu: nhiều xe máy sát nhau dễ nhập thành một cụm; một xe có thể bị tách cụm; xe đứng lâu có thể bị hấp thụ vào nền. MOG2 không phân loại xe máy/ô tô/bus, nên mAP theo lớp phương tiện của baseline này là `N/A`.

Có khoảng khởi tạo nền được cấu hình và ghi lại; khi so sánh chuỗi thời gian, dùng cùng khoảng thời gian hợp lệ cho các phương pháp. Foreground MOG2 không được dùng làm bộ lọc bắt buộc trước YOLO, vì sẽ loại các xe đang dừng chờ đèn.

### 5.3. YOLOv8 và ROI

Bắt đầu với `yolov8n.pt`, so sánh `yolov8s.pt` nếu phần cứng cho phép. Ultralytics hỗ trợ YOLOv8 detection và segmentation; MVP dùng detection để kiểm soát thời gian triển khai. [Tài liệu YOLOv8](https://docs.ultralytics.com/models/yolov8/).

Suy luận trên ảnh màu đầy đủ hoặc crop chữ nhật bao ROI với phép ánh xạ tọa độ rõ ràng. Lọc detection theo các lớp mục tiêu và điểm giữa cạnh đáy trong ROI. Không đưa ảnh biên nhị phân trực tiếp vào YOLOv8 pretrained.

Với trọng số COCO, đọc tên lớp từ model và giữ mapping tương ứng. Nếu fine-tune trên 4 lớp riêng, tạo mapping riêng trong cấu hình; không tái sử dụng ID COCO một cách mặc định. Chọn confidence/NMS/imgsz trên validation. Thử độ phân giải lớn hơn khi xe máy ở xa quá nhỏ và đo lại tốc độ.

Fine-tune là bước có điều kiện: chỉ thực hiện nếu đã có đủ nhãn và dữ liệu validation cho thấy pretrained bỏ sót xe máy đáng kể. Lưu cấu hình, seed, trọng số và phiên bản thư viện được kiểm thử; không tự nâng phiên bản trong quá trình so sánh.

Tracking là phần mở rộng để đếm qua vạch và đo chuyển động tương đối. Có thể dùng ByteTrack trong chế độ tracking của Ultralytics. Xử lý frame liên tiếp hoặc cấu hình tốc độ lấy mẫu phù hợp; việc trích frame gán nhãn ở 1 Hz không đồng nghĩa chạy tracker ở 1 Hz. [Tracking](https://docs.ultralytics.com/modes/track/).

### 5.4. Giả thuyết nghiên cứu và đối chứng

Giả thuyết 1: chọn ngưỡng biên theo điều kiện sáng giúp đề xuất ROI gần ROI tham chiếu hơn và giảm công chỉnh sửa so với ngưỡng cố định.

Giả thuyết 2: ROI và tiền xử lý được hiệu chỉnh giúp giảm lỗi đếm trong cảnh khó, với chi phí tốc độ chấp nhận được. Không khẳng định kết hợp Canny và YOLO tự thân là đóng góp mới về học thuật hoặc chắc chắn tốt hơn trước khi có số liệu.

| Thí nghiệm | Cấu hình | Chỉ số trọng tâm |
| --- | --- | --- |
| A | ROI thủ công + MOG2 | MAE tổng số cụm so với đếm tay, chiếm dụng foreground, FPS |
| B | YOLOv8 + ROI thủ công, ảnh gốc | mAP, MAE theo lớp, FPS; đối chứng nhận dạng |
| C | YOLOv8 + cùng ROI thủ công + CLAHE | Tác động của CLAHE, riêng ngày/đêm |
| D | Canny ngưỡng cố định → ROI đề xuất | ROI IoU và thời gian chỉnh sửa |
| E | Canny thích nghi → ROI đề xuất | ROI IoU và thời gian chỉnh sửa so với D |
| F | Pipeline hoàn chỉnh với ROI đề xuất đã duyệt | Trải nghiệm từ upload đến báo cáo, chất lượng và tốc độ thực tế |

Giữ nguyên ảnh mẫu và thao tác hậu xử lý giữa D/E; tính ROI IoU **trước** khi chỉnh tay, lưu cả ROI trước/sau. B/C dùng cùng ROI để tách riêng tác động CLAHE. Nếu F cuối cùng dùng ROI giống B/C, không quy cải thiện đếm xe cho tách biên; đóng góp của tách biên lúc đó được đo bằng chất lượng gợi ý và công chỉnh sửa. Thử loại bỏ CLAHE hoặc thay ngưỡng cố định từng thành phần, không thay nhiều biến cùng lúc.

## 6. Kế hoạch 5 tuần

| Tuần | Công việc | Sản phẩm và điều kiện hoàn thành |
| --- | --- | --- |
| 1 — Bài toán, dữ liệu, upload | Chốt lớp xe và cách đếm; khảo sát 3 nút giao; thu clip đầu tiên; xây upload, catalog, metadata, xem trước và danh sách video | Upload ít nhất 2 video thành `data/video-1`, `data/video-2`; mở lại được sau restart; không ghi đè khi trùng tên file |
| 2 — ROI, tiền xử lý, nhãn | Trích frame; xây giao diện vẽ ROI; Canny cố định/thích nghi; CLAHE; hoàn tất quy ước nhãn và chia tập; bắt đầu gán nhãn | Mỗi video dùng thử có ROI hợp lệ; lưu được cấu hình; có nhãn mẫu đã rà soát và danh sách split cố định |
| 3 — Baseline và đo lường | MOG2, morphology, đếm cụm, xuất CSV; chạy YOLOv8 pretrained sớm để phát hiện rủi ro; tiếp tục gán nhãn | Baseline chạy trọn clip ngày/đêm; có kiểm tra đếm tay, log thời gian và lỗi; quyết định có cần fine-tune |
| 4 — Phương pháp đề xuất | Hoàn thiện YOLOv8 + ROI + chiếm dụng; tổng hợp thời gian, biểu đồ và video lớp phủ; fine-tune nếu đủ điều kiện; tracking nếu còn thời gian | Luồng upload → ROI → phân tích → kết quả hoàn chỉnh; nhiều run không ghi đè nhau; tham số được chốt trên validation |
| 5 — Thực nghiệm, báo cáo | Khóa cấu hình; chạy test và các đối chứng; phân tích lỗi xe máy/che khuất/đêm; đo FPS; hoàn thiện hướng dẫn và demo | Bảng kết quả thật kèm số mẫu/phần cứng; CSV/video minh họa; báo cáo hạn chế và khả năng hỗ trợ mô phỏng đèn |

Ưu tiên thực hiện: **upload/lưu trữ → ROI thủ công → YOLOv8 đếm xe → chiếm dụng/CSV → baseline → biên thích nghi và đối chứng → báo cáo**. Mỗi tuần vẫn bám nội dung nghiên cứu ở bảng; thứ tự ưu tiên giúp giữ được sản phẩm chạy đầy đủ nếu thời gian bị thiếu. Không hy sinh ground truth và đánh giá để thêm tính năng phụ.

## 7. Giao diện và công nghệ đề xuất

Chọn ứng dụng Python cục bộ với Streamlit để phù hợp đồ án 5 tuần; thuật toán tách thành module để sau này chuyển sang FastAPI nếu cần nhiều người dùng. Đây là lựa chọn thiết kế, chưa phải phụ thuộc đã cài. Kiểm chứng thành phần vẽ đa giác trên trình duyệt ngay tuần 2 trước khi khóa thư viện giao diện.

| Thành phần | Công nghệ / trách nhiệm |
| --- | --- |
| Giao diện | Streamlit: upload, danh sách video, cấu hình, trạng thái, biểu đồ |
| Video và ảnh | OpenCV, NumPy; FFmpeg khi cần chuyển mã để phát video trong trình duyệt |
| Nhận dạng | Ultralytics YOLOv8, PyTorch; CPU trước, CUDA nếu máy hỗ trợ |
| Lưu trữ | File theo từng video + SQLite cấp ID và chỉ mục |
| Báo cáo | pandas, Plotly hoặc Matplotlib; CSV/JSON và MP4 |
| Tác vụ dài | Worker cục bộ, một run hoạt động tại một thời điểm trong MVP; tiến độ ghi vào catalog |

Ứng dụng gồm bốn màn hình: danh sách/upload video; xem trước và cấu hình ROI; chạy phân tích có tiến độ; xem kết quả và so sánh run. Nhãn thao tác hiển thị bằng tiếng Việt.

Không chạy lại inference chỉ vì giao diện làm mới. Worker nhận đường dẫn video và cấu hình đã lưu, hỗ trợ báo lỗi, hủy giữa chừng và chạy lại từ đầu. Video mới không cần upload lại để thử cấu hình khác.

Ví dụ cột trong `timeseries.csv`:

```text
video_id,run_id,roi_id,t_start_s,t_end_s,n_samples,n_motorcycle_mean,n_car_mean,n_bus_mean,n_truck_mean,n_total_mean,occupancy_mean,density_level,valid_fraction
```

`detections.csv` lưu từng detection với `frame_index`, `timestamp_s`, lớp, confidence, box, ROI và `track_id` nếu có. Timestamp dựa trên thời gian video; không lấy thời gian chạy mô hình làm thời gian giao thông. Với video FPS biến thiên, dùng timestamp giải mã thay vì mặc định `frame_index/fps`.

## 8. Đánh giá và tiêu chí nghiệm thu

| Nội dung | Cách đánh giá |
| --- | --- |
| Nhận dạng | mAP@0.5 chung và theo lớp, đặc biệt xe máy; ghi số mẫu ngày/đêm. Dùng nhãn ground truth, không suy ra mAP từ confidence |
| Đếm xe | `MAE = mean(abs(N_pred - N_gt))`, tính tại đúng timestamp/ROI; báo cả tổng và từng lớp, trung bình theo video rồi theo điều kiện |
| ROI | IoU ROI đề xuất với ROI tham chiếu, tỷ lệ phải sửa thủ công và thời gian chỉnh sửa |
| Mức mật độ | Confusion matrix và macro-F1 trên nhãn độc lập; ngưỡng học từ validation rồi giữ nguyên trên test |
| Tốc độ | FPS xử lý và FPS toàn pipeline, thời gian toàn video, độ trễ trung vị/p95; so sánh ngày/đêm trên cùng máy, cấu hình, độ phân giải và stride |
| Ổn định | Độ lệch chuẩn hoặc hệ số biến thiên FPS theo cửa sổ; ghi frame lỗi, đoạn thiếu dữ liệu và thời gian khởi tạo |

Ultralytics cung cấp đánh giá mAP qua chế độ validation; mAP@0.5 dùng ngưỡng IoU 0.5. Báo rõ đang đánh giá toàn frame hay trong ROI. [Validation](https://docs.ultralytics.com/modes/val/).

Đề xuất dùng mAP trên toàn frame có nhãn đầy đủ cho chất lượng detector, và MAE trong ROI cho bài toán ứng dụng. Nếu thêm mAP trong ROI, phải dùng cùng ROI tham chiếu và quy tắc lọc cho prediction/ground truth, báo tên riêng để không lẫn với mAP toàn frame.

Đo tốc độ sau bước warm-up, báo riêng thời gian tải model. FPS inference không bao gồm xuất video phải được phân biệt với FPS toàn pipeline bao gồm decode, tiền xử lý, nhận dạng, tính chỉ số và ghi kết quả. Chạy ba lần cho cấu hình chính; đồng bộ GPU khi đo nếu dùng CUDA. Video xử lý với stride phải ghi rõ tỷ lệ lấy mẫu và coverage, không báo tốc độ bỏ frame như tốc độ xử lý mọi frame.

Kiểm tra chức năng bắt buộc khi triển khai:

- Hai video upload tạo hai hồ sơ độc lập; video gốc được giữ nguyên; restart vẫn đọc lại được.
- Upload đồng thời, upload lặp do giao diện làm mới và tên file trùng không làm ghi đè dữ liệu.
- File hỏng, codec không hỗ trợ, hết dung lượng hoặc run bị ngắt có trạng thái lỗi rõ ràng; không báo hoàn thành giả.
- ROI phải có ít nhất ba đỉnh hợp lệ, không tự cắt, diện tích dương; tọa độ vẫn đúng khi ảnh xem trước bị resize hoặc video có metadata xoay.
- Box chồng nhau không làm `occupancy > 1`; frame hợp lệ không có xe trả về 0; frame thiếu dữ liệu trả về missing.
- Phân tích lại tạo run mới; model, cấu hình và ROI dùng cho kết quả cũ vẫn truy xuất được.
- Mỗi video hoàn thành có kết quả thời gian, summary và video lớp phủ phát được; bảng mAP/MAE chỉ xuất khi có nhãn phù hợp.

Không cam kết sẵn mAP, MAE hay FPS khi chưa biết chất lượng dữ liệu và phần cứng. Cuối tuần 3 đặt mục tiêu cụ thể từ baseline/validation, rồi báo kết quả test kể cả khi phương pháp đề xuất không tốt hơn. Với nhóm ngày/đêm ít dữ liệu, nêu hạn chế thay vì kết luận tổng quát.

## 9. Sản phẩm bàn giao cuối đồ án

1. Mã nguồn ứng dụng và hướng dẫn cài/chạy với phiên bản phụ thuộc đã khóa.
2. Bộ video tổ chức đúng `data/video-N`, metadata, ROI, nhãn và danh sách train/val/test.
3. Trọng số sử dụng, cấu hình, log và kết quả riêng từng run.
4. Bảng so sánh MOG2, YOLOv8, CLAHE và tách biên cố định/thích nghi, kèm phân tích ngày/đêm.
5. Báo cáo học thuật, ảnh minh họa lỗi và video demo luồng upload đến xuất kết quả.

Bước triển khai đầu tiên sau kế hoạch: dựng module lưu trữ và màn hình upload/danh sách, kiểm chứng đường dẫn `data/video-1`, `data/video-2` trước khi tích hợp các thuật toán thị giác máy tính.
