# Quy ước gán nhãn

## Bounding box

- Dùng bốn lớp: `motorcycle`, `car`, `bus`, `truck`.
- Gán một box cho phương tiện nhìn thấy; không tạo box xe riêng cho người lái.
- Box ôm phần phương tiện quan sát được. Không suy đoán xe bị che hoàn toàn.
- Phương tiện được tính trong ROI khi điểm giữa cạnh đáy của box nằm trong ROI.
- Các frame từ cùng video hoặc phiên quay phải nằm trong cùng train/val/test split.

## counts.csv

```csv
timestamp_s,n_total,n_motorcycle,n_car,n_bus,n_truck
0,12,9,2,0,1
1,13,10,2,0,1
```

Timestamp là thời gian của video. Các cột theo lớp có thể bỏ qua, nhưng
`timestamp_s` và `n_total` là bắt buộc.

## density.csv

```csv
timestamp_s,density_level
0,Thấp
1,Trung bình
```

Chỉ dùng ba nhãn `Thấp`, `Trung bình`, `Cao`. Người gán nhãn không xem kết quả
thuật toán trong lúc tạo ground truth.

## ROI tham chiếu

Sao chép cấu trúc của `roi.json` thành `annotations/roi_reference.json` và vẽ
độc lập với ROI do thuật toán gợi ý. `roi_iou` được tính giữa ROI gợi ý trước
khi sửa tay và ROI tham chiếu.

