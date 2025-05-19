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
git clone https://github.com/minhvuAI2003/PBL5_App.git
cd PBL5_App
pip install -r requirements.txt
```

2. **Clone YOLOv5:**

```bash
git clone https://github.com/ultralytics/yolov5.git
```

3. **Tải trọng số YOLOv5:**

- Đặt file trọng số (ví dụ: `best_2.pt`) vào thư mục `weights/` (tạo mới nếu chưa có).

4. **Kiểm tra lại cấu trúc thư mục:**

PBL5_App/
├── traffic_sign_detect_v5.py         # File chính chạy ứng dụng giao diện
├── yolov5/                           # Thư mục mã nguồn YOLOv5
│   ├── models/
│   ├── utils/
│   └── ...
├── weights/                          # Thư mục chứa file trọng số mô hình
│   └── best_2.pt
├── recordings/                       # Thư mục lưu các video đã ghi lại
│   └── detection_<ngày>_<giờ>.mp4
├── sign_statistics.csv               # File lưu báo cáo các biển báo đã phát hiện
├── requirements.txt                  # Danh sách thư viện cần cài đặt
└── README.md 

## Chạy ứng dụng

```bash
python traffic_sign_detect_v5.py
```

- Chọn nguồn video (Camera hoặc Video File).
- Nhấn **Start** để bắt đầu nhận diện.
- Nhấn **Start Recording** để bắt đầu ghi lại video nhận diện.
- Nhấn **Stop Recording** để dừng ghi hình.
- Nhấn **Stop** để dừng nhận diện và dừng video.
- Xem thống kê và cảnh báo biển báo quan trọng ở panel bên phải.

## Xem lại kết quả và báo cáo

- **Báo cáo biển báo đã phát hiện:**
  - Tất cả các biển báo đã nhận diện sẽ được lưu vào file `sign_statistics.csv` trong thư mục gốc dự án. Bạn có thể mở file này bằng Excel hoặc bất kỳ phần mềm bảng tính nào để xem chi tiết thời gian, loại biển báo và số lượng.

- **Xem lại video đã ghi:**
  - Các video quá trình nhận diện được lưu tự động trong thư mục `recordings/`.
  - Mỗi file video sẽ có tên dạng `detection_<ngày>_<giờ>.mp4`.

- **Mã nguồn xử lý lưu báo cáo và video:**
  - Xem chi tiết cách lưu báo cáo và video trong file `traffic_sign_detect_v5.py`.

## Lưu ý
- Nếu gặp lỗi về PyQt5 plugin, thử cài đặt lại PyQt5 hoặc kiểm tra biến môi trường `QT_QPA_PLATFORM_PLUGIN_PATH`.


---

**Chúc bạn sử dụng thành công!** 
