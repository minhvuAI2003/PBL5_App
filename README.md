# Traffic Sign Detection App

Ứng dụng nhận diện biển báo giao thông thời gian thực sử dụng YOLOv5 và giao diện PyQt5.

## Tính năng

- Nhận diện biển báo giao thông từ camera hoặc video file.
- Thống kê số lượng từng loại biển báo.
- Cảnh báo nổi bật các biển báo quan trọng (có phát âm thanh).
- Ghi lại video quá trình nhận diện.
- Giao diện hiện đại, dễ sử dụng.

## Yêu cầu hệ thống

- Python 3.7+
- pip
- (Tùy chọn) GPU + CUDA để tăng tốc nhận diện

## Cài đặt

1. **Clone dự án và cài đặt các thư viện cần thiết:**

```bash
git clone <link-repo-cua-ban>
cd <ten-thu-muc-du-an>
pip install -r requirements.txt
```

2. **Clone YOLOv5:**

```bash
git clone https://github.com/ultralytics/yolov5.git
```

3. **Tải trọng số YOLOv5:**

- Đặt file trọng số (ví dụ: `best_2.pt`) vào thư mục `weights/` (tạo mới nếu chưa có).
- Bạn có thể huấn luyện hoặc tải sẵn từ [YOLOv5 releases](https://github.com/ultralytics/yolov5/releases).

4. **Kiểm tra lại cấu trúc thư mục:**

```
<ten-thu-muc-du-an>/
├── traffic_sign_detect_v5.py
├── yolov5/
│   ├── models/
│   ├── utils/
│   └── ...
├── weights/
│   └── best_2.pt
└── requirements.txt
```

## Chạy ứng dụng

```bash
python traffic_sign_detect_v5.py
```

- Chọn nguồn video (Camera hoặc Video File).
- Nhấn **Start** để bắt đầu nhận diện.
- Nhấn **Start Recording** để ghi lại video nhận diện.
- Xem thống kê và cảnh báo biển báo quan trọng ở panel bên phải.

## Lưu ý

- Nếu chạy trên Mac M1/M2, hãy đảm bảo cài đúng phiên bản torch cho ARM.
- Nếu gặp lỗi về PyQt5 plugin, thử cài đặt lại PyQt5 hoặc kiểm tra biến môi trường `QT_QPA_PLATFORM_PLUGIN_PATH`.



---

**Chúc bạn sử dụng thành công!** 