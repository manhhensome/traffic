# Hướng dẫn chạy project từ đầu đến cuối

Tài liệu dành cho project `C:\Users\admin\PycharmProjects\traffic`, chạy trên Windows bằng PowerShell hoặc Terminal của PyCharm. Các bước dưới đây được đối chiếu trực tiếp với mã nguồn, giao diện và các định dạng output hiện có ngày 08/10/2026.

Mục tiêu: mở ứng dụng → nhập video → chọn vùng mặt đường (ROI) → chạy MOG2/YOLOv8 → xem và xuất kết quả. Phần đánh giá và fine-tune ở cuối chỉ thực hiện khi có nhãn thật.

## Bắt đầu nhanh

Từ thư mục gốc project, ba lệnh tối thiểu để mở ứng dụng là:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Sau đó mở `http://localhost:8501`, lưu video, lưu ROI và chạy thử YOLO hoặc MOG2 với giới hạn 30 giây. Các phần sau giải thích đầy đủ ý nghĩa, dữ liệu đầu vào/đầu ra và quy trình đánh giá đúng.

## Kiến trúc và phạm vi thực tế

Ứng dụng là một giao diện Streamlit; không có REST API hay lệnh CLI riêng để chạy phân tích video. Nút **Chạy phân tích** tạo một worker nền (`worker.py`) để tránh khóa giao diện. Không chạy `worker.py` thủ công, vì worker cần tệp job do ứng dụng tạo.

| Thành phần | Vai trò |
| --- | --- |
| `app.py` | Giao diện upload, metadata, ROI, cấu hình, kết quả và báo cáo tổng hợp |
| `src/storage.py` | Lưu video, metadata và catalog SQLite; chống upload trùng bằng SHA-256 |
| `src/preprocessing.py`, `src/roi.py` | CLAHE, Canny/Sobel, gợi ý và kiểm tra ROI |
| `src/analysis.py` | Pipeline đọc video, YOLO/MOG2, tính mật độ, ghi toàn bộ artifact của một run |
| `worker.py`, `src/jobs.py` | Theo dõi và hủy job nền; toàn project chỉ có một job đang hoạt động |
| `src/evaluation.py` | MAE, Macro-F1, ROI IoU và mAP khi có nhãn tương ứng |
| `src/reporting.py` | Gom các run hoàn tất thành hai CSV trong `reports/` |
| `train.py` | Fine-tune YOLOv8 từ dataset YOLO đã chuẩn bị |

YOLO chỉ giữ bốn lớp có tên `motorcycle`, `car`, `bus`, `truck`. MOG2 chỉ tìm cụm chuyển động (`moving_object`), nên không phải mô hình nhận diện hoặc phân loại xe. Cả hai phương pháp đều chỉ giữ đối tượng có **điểm giữa cạnh đáy bounding box** nằm trong ROI.

## 1. Lộ trình thực hiện

| Bước | Việc cần làm | Điều kiện hoàn thành |
| --- | --- | --- |
| 1 | Chuẩn bị Python, môi trường và thư viện | Import được các thư viện, `pip check` không báo xung đột |
| 2 | Kiểm thử và mở Streamlit | Test kết thúc `OK`, giao diện mở được |
| 3 | Upload video, nhập metadata | Video xuất hiện trong danh sách và có thông tin kỹ thuật |
| 4 | Chọn và lưu ROI | Vùng ROI đúng mặt đường, không tự cắt |
| 5 | Chạy thử MOG2 và YOLOv8 trong 30 giây video | Hai lần chạy hoàn tất, có CSV và video kết quả |
| 6 | Chạy toàn bộ video | Run có trạng thái `completed`, kết quả đủ phạm vi mong muốn |
| 7 | Xuất và so sánh kết quả | Có báo cáo tổng hợp các run cần dùng |
| 8 | Đánh giá bằng nhãn thật, nếu làm đồ án | Có metric tương ứng với nhãn, tách train/val/test đúng |
| 9 | Lưu dữ liệu và đóng ứng dụng | Không còn job đang xử lý, đã sao lưu kết quả |

## 2. Chuẩn bị môi trường

Chuẩn bị một video giao thông có camera cố định, thấy rõ mặt đường. Có thể dùng clip ngắn để thử; clip 5–10 phút dùng cho thí nghiệm đầy đủ. Ứng dụng nhận MP4, MOV, AVI, MKV, WEBM và M4V, giới hạn 1024 MB/file trong `.streamlit/config.toml`. Dành dung lượng trống cho video gốc, các video kết quả và frame trích xuất.

Mở PowerShell tại thư mục gốc project. Nếu Terminal đang ở thư mục cha, dùng đường dẫn tuyệt đối (chỉ dùng lệnh này khi cần đổi thư mục):

```powershell
Set-Location 'C:\Users\admin\PycharmProjects\traffic'
```

Kiểm tra môi trường đã có:

```powershell
Test-Path .\.venv\Scripts\python.exe
.\.venv\Scripts\python.exe --version
```

Nếu đã có `.venv` và lệnh phiên bản chạy được, dùng tiếp môi trường đó. Bản workspace được kiểm tra có `.venv/pyvenv.cfg` ghi Python 3.14.5; đây là thông tin cấu hình, không phải yêu cầu bắt buộc về phiên bản.

**Chỉ khi chưa có `.venv`**, dùng Python đã cài để tạo. Ưu tiên Python 3.10–3.14 64-bit; bộ thư viện thực tế được quyết định bởi `requirements.txt` và thông báo của `pip`.

```powershell
python --version
python -m venv .venv
```

Nếu máy chỉ nhận lệnh `py`, thay `python` bằng `py` trong hai lệnh trên. Nếu cả hai lệnh đều không tồn tại, cài Python và mở lại Terminal trước khi tiếp tục. Không tạo đè môi trường đang lỗi; kiểm tra interpreter và quyền chạy trước.

Cài thư viện bằng đúng Python của project:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import streamlit, cv2, numpy, pandas, yaml, psutil, ultralytics, torch; print('Import OK'); print('CUDA:', torch.cuda.is_available())"
```

Cần Internet để cài các gói chưa có. Hướng dẫn sử dụng đường dẫn Python trực tiếp nên không cần chạy `Activate.ps1` hoặc đổi execution policy của PowerShell.

Nếu dùng PyCharm, chọn interpreter của project là:

```text
.\.\traffic\.venv\Scripts\python.exe
```

Kiểm tra trọng số:

```powershell
Test-Path .\models\yolov8n.pt
```

Workspace hiện có file này. Nếu thiếu, lần khởi tạo YOLO có thể cần tải trọng số qua Internet. MOG2 không dùng trọng số YOLO. Chọn `cpu` cho lần đầu; chỉ chọn `0 (CUDA)` khi phép kiểm tra CUDA trả về `True`.

## 3. Kiểm thử và mở ứng dụng

Chạy bộ kiểm thử sẵn có từ thư mục gốc project:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kết quả mong đợi là `OK`. Nếu có lỗi, đọc lỗi đầu tiên để xử lý môi trường hoặc mã nguồn trước khi chạy thí nghiệm dài. Test có kiểm tra pipeline MOG2 với video tổng hợp và worker nền; vượt qua test chưa chứng minh độ chính xác YOLO trên video giao thông thật.

Mở ứng dụng:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Giữ Terminal này hoạt động. Mở địa chỉ `Local URL` mà Streamlit in ra, thường là `http://localhost:8501`. Không dùng `python app.py` để khởi chạy giao diện.

Nếu cổng đang bị dùng, chọn cổng khác:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

**Lưu ý cấu hình:** giao diện hiện lấy tham số từ các điều khiển trong `app.py`, chưa đọc `configs/default.yaml`. Muốn đổi một lần chạy, chỉnh trực tiếp ở tab **Phân tích**. Mỗi run tự lưu cấu hình thực tế để kiểm tra lại.

## 4. Nhập video và bối cảnh ghi hình

1. Trong thanh bên, bấm **Chọn video từ máy tính**.
2. Chọn một hoặc nhiều file rồi bấm **Lưu video đã chọn**.
3. Chọn **Video đang làm việc**.
4. Trong tab **Video**, kiểm tra khung hình, FPS và thời lượng.
5. Điền **Nút giao**, **Camera ID**, **Mã phiên quay**, điều kiện sáng, thời tiết, thời điểm và nguồn ghi hình; bấm **Lưu metadata**.

Ví dụ mã phiên quay: `nga-tu-A_20261002_sang`. Các clip cắt từ cùng một phiên phải có cùng mã này để chia tập dữ liệu đúng.

Mỗi video mới có thư mục `data/video-N/`. File giống hệt video đã lưu được nhận diện bằng hash và dùng lại hồ sơ cũ. Cảnh báo clip ngoài 5–10 phút không ngăn việc chạy thử.

Kết quả cần có: video xuất hiện trong danh sách, thông tin đúng, có video gốc và `metadata.json` trong thư mục tương ứng.

## 5. Chọn vùng mặt đường

1. Mở tab **ROI & tách biên**, chọn thời điểm có thể nhìn rõ đường.
2. Chọn **Canny thích nghi**, **Canny cố định** hoặc **Sobel** để xem biên. Có thể bật **Áp dụng CLAHE** để xem ảnh tăng tương phản.
3. Bấm **Gợi ý lại ROI** nếu muốn lưu đề xuất của thuật toán trước khi chỉnh tay.
4. Chỉnh `X1/Y1` đến `X4/Y4` theo thứ tự quanh biên đa giác, ví dụ trên trái → trên phải → dưới phải → dưới trái.
5. Kiểm tra lớp phủ: ROI bao vùng đường cần đo, hạn chế vỉa hè và nền ngoài đường. Không đặt các cạnh tự cắt hoặc các điểm trùng nhau.
6. Bấm **Lưu ROI** và chờ thông báo thành công.

Giao diện hiện chỉnh một ROI bốn đỉnh cho mỗi video. Các tọa độ nhập là pixel; file ROI lưu tọa độ chuẩn hóa. Run sẽ giữ bản sao ROI đã dùng.

CLAHE ở tab xem trước và **CLAHE trước khi nhận dạng** ở tab phân tích là hai lựa chọn riêng. Chọn Sobel/Canny phục vụ xem biên và hỗ trợ ROI; không thay thế bước nhận dạng YOLO. Trong mã hiện tại, nút gợi ý ROI dùng Canny cố định hoặc thích nghi, kể cả khi đang xem Sobel.

Kết quả cần có: `data/video-N/roi.json` và không còn cảnh báo yêu cầu lưu ROI trong tab **Phân tích**.

## 6. Chạy thử trên 30 giây đầu video

30 giây ở đây là thời lượng video được phân tích, không phải thời gian chờ xử lý.

Lần 1: chọn **MOG2 — baseline chuyển động** để kiểm tra luồng xử lý. Lần 2: chọn **YOLOv8 — nhận dạng phương tiện** để kiểm tra mô hình. Giữ cùng video và ROI cho cả hai lần.

| Tham số | Giá trị chạy thử |
| --- | --- |
| Phân tích mỗi N frame | 5 |
| Giới hạn thời lượng | 30 giây |
| Thiết bị YOLO | `cpu` |
| Ngưỡng confidence YOLO | 0.35 |
| Kích thước ảnh YOLO | 640 |
| CLAHE trước khi nhận dạng | Tắt |
| Khởi tạo nền MOG2 | 2 giây |
| Ngưỡng mật độ trung bình / cao | 0.12 / 0.28 |
| Biên hysteresis mật độ | 0.02 |

1. Bấm **Chạy phân tích**.
2. Dùng **Làm mới trạng thái** để xem tiến độ. Có thể chuyển tab khi worker chạy nền.
3. Chờ trạng thái `completed` rồi mở tab **Kết quả**, chọn đúng **Lần chạy**.
4. Kiểm tra video phủ kết quả, biểu đồ số xe/vật thể, chiếm dụng và số frame đã lấy mẫu.
5. Chuyển phương pháp và lặp lại để có hai run riêng.

Ứng dụng chỉ cho phép một job xử lý tại một thời điểm trên toàn bộ kho video. Nếu cần dừng giữa chừng, bấm **Hủy tác vụ**; tác vụ bị hủy được đánh dấu `interrupted`.

MOG2 đếm cụm chuyển động, không phân loại xe. YOLO lọc bốn lớp `motorcycle`, `car`, `bus`, `truck`; một detection được tính khi điểm giữa cạnh đáy box nằm trong ROI. Số bằng 0 có thể là kết quả hợp lệ, nhưng nếu hình rõ nhiều xe thì cần kiểm tra ROI, confidence và chất lượng nhận dạng.

## 7. Chạy toàn bộ video và xuất kết quả

Khi hai lần chạy thử đã thành công:

1. Chọn video và ROI đã duyệt.
2. Đặt **Giới hạn thời lượng = 0** để chạy toàn bộ video.
3. Giữ cấu hình đã chọn rồi bấm **Chạy phân tích**. Giá trị `N=5` vẫn lấy mẫu mỗi năm frame; đặt `N=1` nếu cần xử lý mọi frame và chấp nhận thời gian lâu hơn.
4. Chờ `completed`, mở **Kết quả** và chọn run vừa tạo.
5. Bấm **Tải timeseries.csv**, **Tải detections.csv**, **Tải summary.json** theo nhu cầu.

Các file nằm trong `data/video-N/runs/run-XXXX/`:

| File | Nội dung |
| --- | --- |
| `annotated.mp4` | Video phủ ROI và kết quả xử lý |
| `detections.csv` | Detection/vật thể phát hiện |
| `frame_metrics.csv` | Số đo trên các frame được lấy mẫu |
| `timeseries.csv` | Chuỗi số đo theo thời gian, gồm chiếm dụng làm trơn |
| `summary.json` | Tổng hợp lần chạy |
| `evaluation.json` | Các metric đánh giá khi có nhãn |
| `config.json`, `config.yaml` | Tham số thực tế của run |
| `roi.json`, `roi_mask.png` | ROI sử dụng trong run |
| `runtime.json` | Thông tin môi trường và trọng số |
| `status.json`, `run.log` | Trạng thái và log xử lý |

Số xe là số hiện diện tại thời điểm lấy mẫu; không cộng theo thời gian để kết luận tổng lượt xe. Chiếm dụng là tỷ lệ diện tích ảnh trong ROI, chưa phải mật độ xe/km. Các ngưỡng 0.12 và 0.28 là điểm khởi đầu cần hiệu chỉnh, không phải ngưỡng chuẩn cho mọi camera.

## 8. So sánh thí nghiệm

Trên cùng video, cùng ROI, cùng thời lượng và stride, tạo các run:

| Run đề xuất | Phương pháp | CLAHE |
| --- | --- | --- |
| A | MOG2 | Tắt |
| B | YOLOv8 | Tắt |
| C | YOLOv8 | Bật |

Lặp lại trên video ngày và đêm nếu có. Mỗi lần bấm chạy tạo run mới, không cần upload lại video.

Trong tab **Kết quả**, bấm **Tạo báo cáo tổng hợp**. Báo cáo được lưu tại:

```text
reports/experiment_runs.csv
reports/condition_summary.csv
```

Báo cáo tổng hợp các run hoàn tất, kể cả run thử ngắn. Khi viết báo cáo chính thức, chọn đúng các run có cùng điều kiện so sánh và ghi lại video ID, run ID, ROI, thời lượng, stride cùng cấu hình. Metadata ngày/đêm cần được nhập trước khi chạy.

## 9. Đánh giá bằng nhãn thật — phần tùy chọn

Không cần nhãn để mở app hoặc phân tích video. Cần nhãn để kết luận về độ chính xác.

1. Nhập đúng **Mã phiên quay** cho các video.
2. Mở **Chia tập dữ liệu** ở thanh bên và bấm **Tạo chia 60/20/20 theo phiên quay**. Kiểm tra `data/splits.json`: đây là chia nhóm xấp xỉ; ít phiên có thể khiến một tập không có dữ liệu. Nếu gán thủ công, bảo đảm một phiên quay không xuất hiện trong nhiều tập.
3. Trong tab **Video**, chọn khoảng trích frame rồi bấm **Trích frame để gán nhãn**. File ảnh và `manifest.csv` được lưu ở `data/video-N/frames/`. Dùng timestamp từ manifest làm mốc đối chiếu.
4. Gán nhãn theo [ANNOTATION_GUIDE.md](docs/ANNOTATION_GUIDE.md), đếm trong cùng ROI và cùng mốc thời gian video. Các số trong ví dụ của tài liệu chỉ minh họa định dạng.
5. Trong tab **Kết quả**, upload các nhãn tương ứng và bấm **Lưu các nhãn đã chọn**.
6. Chạy lại phân tích để tạo run mới có đánh giá. Việc lưu nhãn không cập nhật metric của run cũ.

| Nhãn | Vị trí sau khi lưu | Metric |
| --- | --- | --- |
| Đếm tay | `annotations/counts.csv` | MAE tổng số xe và theo lớp nếu có |
| Mức mật độ | `annotations/density.csv` | Macro-F1 và confusion matrix |
| ROI tham chiếu độc lập | `annotations/roi_reference.json` | IoU với ROI gợi ý trước chỉnh tay |
| Bounding box YOLO và cấu hình dataset | `annotations/dataset.yaml` cùng bộ ảnh/nhãn được trỏ tới | mAP trong run YOLO |

Để có ROI IoU, cần bấm **Gợi ý lại ROI** trước đó để tạo `roi_suggestion.json`. ROI tham chiếu phải được vẽ độc lập; không chép nguyên kết quả gợi ý để làm nhãn.

Giao diện chưa có uploader cho dataset bounding box. Cần tự chuẩn bị ảnh/nhãn và đặt `dataset.yaml` tại `data/video-N/annotations/dataset.yaml`; các đường dẫn phải hợp lệ và ID lớp phải khớp mô hình được đánh giá. Không dùng trực tiếp nhãn bốn lớp đánh số lại với trọng số pretrained nếu ánh xạ lớp không khớp. mAP được tính trên dataset ảnh đầy đủ được cấu hình, không chỉ các box trong ROI video.

`splits.json` chỉ lưu phân công video, không tự tạo dataset YOLO hoặc chuyển ảnh/nhãn sang thư mục train/val/test. Hiện lời gọi đánh giá mAP dùng tập validation mặc định của thư viện; cần quy trình đánh giá riêng nếu muốn báo cáo mAP trên test. Chỉ hiệu chỉnh tham số bằng train/val và giữ test độc lập.

Thiếu nhãn thì metric tương ứng có thể là `null`, hoặc trạng thái MAE là `not_evaluated`. Đây không phải lỗi chạy ứng dụng; đọc thêm thông báo trong `evaluation.json`.

## 10. Chuẩn bị dataset YOLO và fine-tune — chỉ khi đã có nhãn

Chỉ thực hiện sau khi dữ liệu và nhãn đã sẵn sàng. Không cần fine-tune để chạy demo. Trích frame trong giao diện chỉ tạo ảnh và `manifest.csv`; nó **không** tự sinh box YOLO, file `.txt`, dataset YAML hoặc split thư mục. Hãy gán nhãn các frame theo [quy ước](docs/ANNOTATION_GUIDE.md), rồi tự tổ chức dataset theo định dạng YOLO, ví dụ:

```text
data/traffic_yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

Mỗi `images/<split>/abc.jpg` cần có `labels/<split>/abc.txt` tương ứng, với mỗi dòng box là `class_id x_center y_center width height`; bốn tọa độ sau đã chuẩn hóa trong khoảng 0–1. Để tương thích detector trong app, dùng đúng mapping sau: `0=motorcycle`, `1=car`, `2=bus`, `3=truck`. Ví dụ `data/traffic_yolo/dataset.yaml`:

```yaml
path: C:/Users/admin/PycharmProjects/traffic/data/traffic_yolo
train: images/train
val: images/val
test: images/test
names:
  0: motorcycle
  1: car
  2: bus
  3: truck
```

Không để frame từ cùng `recording_session_id` ở nhiều split. `data/splits.json` do app tạo chỉ là bản phân công video để kiểm soát rò rỉ dữ liệu; người dùng vẫn phải chép/liên kết ảnh và nhãn vào đúng thư mục train/val/test. Dùng validation để chọn tham số và chỉ xem test sau khi đã chốt mô hình.

Script nhận đường dẫn YAML bất kỳ; ví dụ sau dùng YAML ở trên:

```powershell
.\.venv\Scripts\python.exe train.py --data data\traffic_yolo\dataset.yaml --model models\yolov8n.pt --device cpu --epochs 50 --imgsz 640 --batch 8 --seed 42
```

Nếu CUDA đã hoạt động, có thể dùng `--device 0`; nếu thiếu bộ nhớ, giảm batch. Script lưu kết quả dưới `models/training/`; lấy đường dẫn trọng số thực tế từ log, vì tên thư mục có thể thay đổi giữa các lần huấn luyện.

Huấn luyện lưu artifact vào `models/training/hanoi-traffic*` theo Ultralytics (thường trọng số tốt nhất là `weights/best.pt`). Giao diện hiện chưa có ô chọn model và luôn khởi tạo `AnalysisConfig.model_path` với `models/yolov8n.pt`. Huấn luyện xong không tự đổi mô hình trong app. Muốn dùng trọng số mới, cần thay giá trị `model_path` khi khởi tạo `AnalysisConfig` trong `app.py` hoặc bổ sung ô chọn model vào giao diện; chỉ sửa `configs/default.yaml` chưa có tác dụng. Không ghi đè trọng số pretrained nếu cần giữ baseline để so sánh.

Để tính mAP trong một run YOLO trên app, đặt (hoặc sao chép) YAML hợp lệ tại `data/video-N/annotations/dataset.yaml`. Pipeline gọi validation của Ultralytics trên trường `val` trong YAML, nên mAP trong `evaluation.json` có phạm vi `full_frame_dataset`, không phải chỉ vùng ROI và không tự dùng split trong `data/splits.json`. Muốn báo cáo test cuối cùng, chạy validation riêng với YAML/thiết lập test đã khóa; không coi mAP validation là mAP test.

## 11. Cách đọc output, metric và kết luận

Một run hoàn tất được tạo ở `data/video-N/runs/run-XXXX/`. Run thất bại hoặc bị hủy có thể chỉ có một phần file; chỉ các run với `summary.json.status = "completed"` mới xuất hiện trong tab **Kết quả** và báo cáo tổng hợp.

| Artifact | Cách đọc / mục đích |
| --- | --- |
| `status.json`, `run.log` | Trạng thái cuối và lỗi của run: `queued`, `processing`, `completed`, `failed`, `interrupted` |
| `config.json`, `config.yaml` | Cấu hình thật đã dùng; đây là nguồn đối chiếu thí nghiệm, không phải `configs/default.yaml` |
| `runtime.json` | Python, hệ điều hành, phiên bản thư viện, SHA-256 trọng số và danh sách lớp |
| `roi.json`, `roi_mask.png` | ROI được đóng băng cho riêng run; dùng để kiểm tra tính công bằng giữa các run |
| `annotated.mp4` | Chỉ các frame được lấy mẫu, phủ ROI, box, số đối tượng, chiếm dụng và mức mật độ |
| `detections.csv` | Mỗi box sau lọc ROI: `frame_index`, `timestamp_s`, `class_name`, `confidence`, `x1`, `y1`, `x2`, `y2` |
| `frame_metrics.csv` | Mỗi frame lấy mẫu: số theo lớp, `n_total`, chiếm dụng, latency và cờ `valid` |
| `timeseries.csv` | Tổng hợp theo từng giây: trung bình số xe, chiếm dụng, cửa sổ làm trơn 5 giây, nhãn mật độ, latency |
| `summary.json` | Chỉ số gọn để so sánh tốc độ, số đối tượng, chiếm dụng và phân bố mức mật độ |
| `evaluation.json` | Kết quả so với nhãn thật; `null`/`not_evaluated` nghĩa là chưa có hoặc không ghép được nhãn, không phải độ chính xác bằng 0 |

Giải thích các chỉ số chính:

- `n_total` là **số phương tiện/vật thể đang hiện diện** tại một frame hoặc trung bình trong một giây, không phải tổng lượt xe đã đi qua. Project chưa có tracker/vạch đếm để suy ra lưu lượng.
- `occupancy` là tỷ lệ pixel chiếm trong ROI. Với YOLO, đó là hợp các vùng box xe trong ROI (`bounding_box_union`); với MOG2, đó là foreground mask (`foreground_mask`). Hai số này hữu ích để so sánh xu hướng trong cùng phương pháp/camera, nhưng không phải mật độ vật lý xe/km và không nên coi là tương đương tuyệt đối giữa YOLO và MOG2.
- `occupancy_smooth` là trung bình trượt 5 giây. `density_level` áp ngưỡng thấp/trung bình/cao và hysteresis lên chuỗi làm trơn; `Khởi tạo` là thời gian warm-up MOG2 chưa hợp lệ.
- `processing_fps` là số frame đã lấy mẫu trên giây xử lý; `pipeline_fps` là toàn bộ frame giải mã trên giây. `latency_median_ms` và `latency_p95_ms` phản ánh độ trễ một mẫu. Khi đổi `frame_stride`, phải ghi rõ cả hai FPS để so sánh đúng.
- `mean_vehicle_or_object_count`, `mean_occupancy`, `max_*` tính trên các sample hợp lệ; MOG2 bỏ qua `mog2_warmup_seconds` khi tổng hợp. Kiểm tra `read_failures` lớn hơn 0 như một dấu hiệu video/codec có vấn đề.

Đánh giá chỉ hợp lệ khi nhãn, timestamp và phạm vi video tương ứng với run. Pipeline ghép mốc dự đoán gần nhất trong sai số tối đa `max(0.6, frame_stride / fps)` giây cho MAE đếm; metric không cho biết chất lượng nếu chỉ có rất ít mẫu ghép được, vì vậy cần đọc `matched_samples` trong `evaluation.json`.

| Metric | Cần chuẩn bị | Ý nghĩa và giới hạn |
| --- | --- | --- |
| MAE tổng / từng lớp | `annotations/counts.csv` | Sai số tuyệt đối trung bình giữa đếm tay và `timeseries.csv`; càng thấp càng tốt. Chỉ các cột nhãn hiện diện mới được tính. |
| Macro-F1 mật độ | `annotations/density.csv` | F1 trung bình không trọng số của `Thấp`, `Trung bình`, `Cao`; đọc kèm `density_confusion_matrix`, nhất là khi dữ liệu lệch lớp. |
| ROI IoU | `roi_suggestion.json` và `annotations/roi_reference.json` | IoU giữa ROI **gợi ý trước khi chỉnh tay** và ROI tham chiếu độc lập; không đo chất lượng ROI thủ công đã lưu. |
| mAP@0.5, mAP@0.5:0.95 | `annotations/dataset.yaml` + dataset box YOLO hợp lệ | Chất lượng detection full-frame từ Ultralytics trên tập `val` YAML; càng cao càng tốt. Không có mAP hợp lệ cho MOG2. |

Ví dụ diễn giải đúng: MAE thấp nhưng `matched_samples` ít thì chưa đủ bằng chứng; mAP cao full-frame không bảo đảm số đếm trong ROI chính xác; mật độ `Thấp` có thể chiếm đa số chỉ vì ngưỡng 0.12/0.28 chưa hiệu chỉnh cho camera. Với báo cáo chính thức, giữ cố định video, ROI, thời lượng, frame stride và thiết bị, rồi so sánh cùng một nhóm run.

## 12. Xử lý lỗi thường gặp

| Hiện tượng | Hướng xử lý |
| --- | --- |
| Không tìm thấy Python hoặc `Access is denied` | Thử lệnh `--version` trong Terminal máy; kiểm tra interpreter gốc ghi ở `.venv/pyvenv.cfg` và quyền thực thi. Chưa kết luận môi trường hỏng chỉ từ lỗi quyền truy cập. |
| `No module named ...` | Cài lại `requirements.txt` bằng `.venv\Scripts\python.exe -m pip`, rồi kiểm tra interpreter PyCharm. |
| Pip không tìm được bản phân phối phù hợp | Đọc tên gói bị lỗi và kiểm tra khả năng hỗ trợ phiên bản Python đang dùng trước khi tạo môi trường mới. |
| Không mở được giao diện | Xem lỗi Terminal và đúng `Local URL`; thử cổng 8502 nếu cổng 8501 đang bận. |
| Test worker báo đang có tác vụ khác | Dừng hoặc hủy job phân tích đang chạy từ giao diện rồi chạy lại test. Test worker kiểm tra cơ chế chỉ cho phép một job trên toàn project, nên sẽ chủ động thất bại khi một job thật còn `processing`. |
| Không bấm được Chạy phân tích | Lưu ROI, đặt ngưỡng cao lớn hơn ngưỡng trung bình, chờ hoặc hủy job đang chạy. |
| Lỗi CUDA | Chuyển về `cpu`, kiểm tra lại `torch.cuda.is_available()`. |
| YOLO không tải được trọng số | Kiểm tra `models/yolov8n.pt`, mạng và thông báo trong log; có thể thử MOG2 để kiểm tra phần còn lại. |
| Video không đọc được | Kiểm tra file có giải mã được; chuyển mã sang định dạng phù hợp, không chỉ đổi phần mở rộng. |
| Video kết quả không phát trong trình duyệt | App hiện xuất codec `mp4v`; mở file bằng trình phát trên máy hoặc chuyển một bản sang MP4/H.264. CSV vẫn có thể kiểm tra riêng. |
| Chạy quá chậm | Thử 30 giây trước; tăng stride hoặc giảm kích thước ảnh YOLO, ghi lại thay đổi khi so sánh. |
| Job `failed` hoặc `interrupted` | Đọc `data/video-N/jobs/job-*.json`, file `.log` tương ứng và `runs/run-XXXX/run.log` nếu run đã được tạo; khắc phục rồi chạy lại. |
| Không có MAE/mAP/F1/IoU | Kiểm tra nhãn, đường dẫn, timestamp và phạm vi video đã chạy; lưu nhãn rồi tạo run mới. |

## 13. Kết thúc và chạy lại lần sau

1. Chờ job hoàn tất hoặc bấm **Hủy tác vụ** trước khi dừng server. Worker là tiến trình riêng, đóng tab trình duyệt không phải thao tác hủy job.
2. Sao lưu `data/` gồm cả `catalog.sqlite`, `models/` cần dùng và `reports/`. Giữ mã nguồn, `requirements.txt`, cấu hình từng run và `runtime.json` để đối chiếu môi trường.
3. Trong Terminal chạy Streamlit, nhấn `Ctrl+C`.
4. Lần sau mở lại bằng hai lệnh:

```powershell
Set-Location 'C:\Users\admin\PycharmProjects\traffic'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Video và các run đã lưu có thể chọn lại từ giao diện. Job bị gián đoạn cần được chạy lại, không tự tiếp tục từ frame cuối.

**Tiêu chí hoàn tất demo:** mở được ứng dụng, upload được video, lưu ROI, chạy YOLO thành công, xem được số liệu và xuất được CSV/JSON. **Tiêu chí hoàn tất thí nghiệm:** bổ sung nhãn thật, đánh giá đúng tập dữ liệu, so sánh các run tương đương và lưu đủ cấu hình/kết quả.

Ghi nhận kiểm tra khi viết hướng dẫn: đã đối chiếu các nút, tham số và đường dẫn với mã nguồn. Lệnh kiểm tra import bằng `.venv\Scripts\python.exe` trong phiên trợ lý trả về `Access is denied`, nên chưa xác nhận chạy ứng dụng hoặc bộ test trong phiên này. Các lệnh trên là quy trình để thực hiện và kiểm chứng trên máy.
