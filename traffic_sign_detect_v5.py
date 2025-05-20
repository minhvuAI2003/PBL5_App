# Import các thư viện chuẩn của Python
import os
import sys
import csv
import time
import threading
from datetime import datetime
from collections import defaultdict

# Import các thư viện bên thứ ba
import cv2
import torch
import numpy as np
import pyttsx3
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QComboBox, 
                           QFileDialog, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# Thử import playsound để phát âm thanh
try:
    from playsound import playsound
    HAS_PLAYSOUND = True
except ImportError:
    print("Warning: playsound module not found. Audio playback will be disabled.")
    print("To enable audio playback, install playsound: pip install playsound==1.2.2")
    HAS_PLAYSOUND = False

# Tạo các thư mục cần thiết để lưu trữ
os.makedirs('recordings', exist_ok=True)  # Thư mục lưu video ghi lại
os.makedirs('sounds_wav', exist_ok=True)  # Thư mục chứa file âm thanh cảnh báo

# Cấu hình đường dẫn plugin cho PyQt5
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.path.dirname(sys.modules["PyQt5"].__file__), "Qt5", "plugins")

# Thêm yolov5 vào đường dẫn Python và import các module cần thiết
sys.path.append('yolov5')
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device

def letterbox_fixed(img, new_shape=(320, 320), color=(114, 114, 114)):
    """
    Thay đổi kích thước ảnh và thêm padding để giữ tỷ lệ khung hình.
    
    Tham số:
        img: Ảnh đầu vào
        new_shape: Kích thước mới (chiều rộng, chiều cao)
        color: Màu padding (BGR)
    
    Trả về:
        Ảnh đã resize với padding, tỷ lệ scale, và giá trị padding
    """
    shape = img.shape[:2]  # chiều cao, chiều rộng
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh

def play_sound_file(audio_path):
    """
    Phát file âm thanh một cách an toàn với xử lý lỗi.
    
    Tham số:
        audio_path: Đường dẫn đến file âm thanh
    """
    if not HAS_PLAYSOUND:
        return
        
    try:
        # Chuyển đổi thành đường dẫn tuyệt đối và xóa tiền tố file://
        audio_path = os.path.abspath(audio_path)
        if audio_path.startswith('file:'):
            audio_path = audio_path[7:]
        elif audio_path.startswith('file:///'):
            audio_path = audio_path[8:]
            
        # Kiểm tra file có tồn tại không
        if not os.path.exists(audio_path):
            print(f"Warning: Sound file not found: {audio_path}")
            return
            
        # Trên macOS, sử dụng afplay thay vì playsound để tương thích tốt hơn
        if sys.platform == 'darwin':
            import subprocess
            subprocess.Popen(['afplay', audio_path])
        else:
            # Sử dụng chuỗi thô để tránh vấn đề với đường dẫn
            audio_path = str(audio_path)
            print(f"Playing sound: {audio_path}")
            threading.Thread(target=playsound, args=(audio_path,), daemon=True).start()
    except Exception as e:
        print(f"Error playing sound: {str(e)}")

def create_multitracker():
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'MultiTracker_create'):
        return cv2.legacy.MultiTracker_create()
    elif hasattr(cv2, 'MultiTracker_create'):
        return cv2.MultiTracker_create()
    else:
        raise RuntimeError("OpenCV is missing MultiTracker_create. Please install opencv-contrib-python.")

def create_tracker():
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
        return cv2.legacy.TrackerKCF_create()
    elif hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    else:
        raise RuntimeError("OpenCV is missing TrackerKCF_create. Please install opencv-contrib-python.")

class TrafficSignDetector(QMainWindow):
    """
    Cửa sổ chính của ứng dụng nhận diện biển báo giao thông.
    
    Lớp này triển khai một ứng dụng GUI có thể:
    - Nhận diện biển báo giao thông theo thời gian thực từ camera hoặc file video
    - Hiển thị kết quả nhận diện với khung bao quanh
    - Phát âm thanh cảnh báo cho các biển báo quan trọng
    - Ghi lại video với các nhận diện
    - Hiển thị thống kê và cảnh báo
    """
    
    def __init__(self):
        """Khởi tạo cửa sổ ứng dụng và các thành phần của nó."""
        super().__init__()
        self.setWindowTitle("Traffic Sign Detection")
        self.setGeometry(100, 100, 1400, 800)
        
        # Khởi tạo engine text-to-speech cho cảnh báo bằng giọng nói
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Tốc độ nói
        self.engine.setProperty('volume', 1.0)  # Âm lượng
        
        # Khởi tạo model YOLOv5 cho việc nhận diện biển báo
        self.device = select_device('')  # Sử dụng CPU hoặc GPU
        self.model = DetectMultiBackend('weights/best_2.pt', device=self.device)
        self.stride = self.model.stride
        self.imgsz = check_img_size((640, 640), s=self.stride)
        
        # Khởi tạo các thành phần xử lý video
        self.cap = None  # Đối tượng bắt video
        self.video_writer = None  # Đối tượng ghi video
        self.timer = QTimer()  # Bộ đếm thời gian cho việc cập nhật frame
        self.timer.timeout.connect(self.update_frame)
        self.is_paused = False  # Trạng thái tạm dừng
        
        # Khởi tạo theo dõi thống kê
        self.sign_stats = defaultdict(int)  # Đếm số lượng mỗi loại biển báo
        self.sign_positions = defaultdict(list)  # Vị trí các biển báo đã phát hiện
        self.played_labels = {}  # Theo dõi thời gian phát âm thanh cuối cùng của mỗi biển báo
        self.cooldown = 5  # Thời gian chờ giữa các lần phát âm thanh (giây)
        
        # Mapping tên class của model với tên file âm thanh
        self.class_to_sound = {
            'cam re trai': 'Cam re trai.wav',
            'cam re phai': 'Cam re phai.wav',
            'cam do xe': 'Cam do xe.wav',
            'cam do xe ngay chan': 'Cam do xe ngay chan.wav',
            'cam do xe ngay le': 'Cam do xe ngay le.wav',
            'cam dung xe va do xe': 'Cam dung xe và do xe.wav',
            'cam oto re trai': 'Cam oto re trai.wav',
            'cam bam coi': 'Cam bam coi.wav',
            'cam xe may 2 banh': 'Cam xe may 2 banh.wav',
            'cam di nguoc chieu': 'Cam di nguoc chieu.wav',
            'toc do toi da 40 km/h': '40kmh.wav',
            'toc do toi da 60 km/h': '60kmh.wav',
            'toc do toi da 80 km/h': '80kmh.wav',
            'cam quay dau': 'Cam quay dau.wav',
            'tre em qua duong': 'Tre em qua duong.wav',
            'duong giao nhau': 'Duong giao nhau.wav',
            'giao nhau voi duong uu tien': 'Giao nhau voi duong uu tien.wav',
            'giao nhau voi duong khong uu tien': 'Giao nhau voi duong khong uu tien.wav',
            'noi giao nhau theo vong xuyen': 'Noi giao nhau theo vong xuyen.wav',
            'cho ngoac nguy hiem phia ban trai': 'Cho ngoac nguy hiem phia ban trai.wav',
            'cho ngoac nguy hiem phia ben phai': 'Cho ngoac nguy hiem phia ben phai.wav',
            'duong nguoi di bo cat ngang': 'Duong nguoi di bo cat ngang.wav',
            'huong di thang phai theo': 'Huong di thang phai theo.wav'
        }
        
        # Khởi tạo giao diện người dùng
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        # Apply dark theme and modern styles
        self.setStyleSheet('''
            QMainWindow {
                background-color: #23272e;
            }
            QLabel, QTextEdit, QComboBox {
                color: #f8f8f2;
                font-size: 16px;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            QLabel#TitleLabel {
                font-size: 28px;
                font-weight: bold;
                color: #50fa7b;
            }
            QPushButton {
                background-color: #44475a;
                color: #f8f8f2;
                border-radius: 8px;
                padding: 8px 18px;
                font-size: 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #6272a4;
            }
            QComboBox, QTextEdit {
                background-color: #282a36;
                border-radius: 6px;
                border: 1px solid #44475a;
            }
            QTextEdit {
                font-size: 15px;
            }
            QGroupBox {
                border: 2px solid #44475a;
                border-radius: 10px;
                margin-top: 10px;
            }
        ''')
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # Left panel for video display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet('''
            background-color: #191a21;
            border-radius: 16px;
            border: 2px solid #44475a;
        ''')
        left_layout.addWidget(self.video_label)
        
        # Video info label
        self.info_label = QLabel("Video Info: Not started")
        self.info_label.setStyleSheet('font-size: 15px; color: #8be9fd;')
        left_layout.addWidget(self.info_label)
        
        # Right panel for controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Title
        title = QLabel("Traffic Sign Detection")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title)
        
        # Source selection
        source_layout = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Camera", "Video File"])
        source_layout.addWidget(QLabel("Source:"))
        source_layout.addWidget(self.source_combo)
        right_layout.addLayout(source_layout)
        
        # Camera selection
        camera_layout = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["0", "1", "2"])
        camera_layout.addWidget(QLabel("Camera:"))
        camera_layout.addWidget(self.camera_combo)
        right_layout.addLayout(camera_layout)
        
        # Control buttons
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_detection)
        right_layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        right_layout.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        right_layout.addWidget(self.stop_button)
        
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        right_layout.addWidget(self.record_button)
        
        # Detection settings
        settings_layout = QHBoxLayout()
        self.conf_thres = QComboBox()
        self.conf_thres.addItems(["0.25", "0.5", "0.75"])
        settings_layout.addWidget(QLabel("Confidence:"))
        settings_layout.addWidget(self.conf_thres)
        right_layout.addLayout(settings_layout)
        
        # Statistics display
        self.stats_label = QLabel("Sign Statistics:")
        self.stats_label.setStyleSheet('font-size: 17px; color: #ffb86c; font-weight: bold;')
        right_layout.addWidget(self.stats_label)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        right_layout.addWidget(self.stats_text)
        
        # Add panels to main layout
        layout.addWidget(left_panel, stretch=2)
        layout.addWidget(right_panel, stretch=1)
        
    def toggle_pause(self):
        if self.is_paused:
            self.is_paused = False
            self.pause_button.setText("Pause")
            self.timer.start(30)
        else:
            self.is_paused = True
            self.pause_button.setText("Resume")
            self.timer.stop()

    def toggle_recording(self):
        if self.video_writer is None:
            try:
                # Create recordings directory if it doesn't exist
                os.makedirs('recordings', exist_ok=True)
                
                # Start recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join('recordings', f"detection_{timestamp}.mp4")
                
                # Use a more compatible codec
                if sys.platform == 'darwin':  # macOS
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
                else:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("Cannot read frame for size detection")
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                height, width = frame.shape[:2]
                
                self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                
                if not self.video_writer.isOpened():
                    raise Exception("Failed to create video writer")
                    
                self.record_button.setText("Stop Recording")
                QMessageBox.information(self, "Recording", f"Started recording to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start recording: {str(e)}")
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
        else:
            try:
                # Stop recording
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                    self.record_button.setText("Start Recording")
                    QMessageBox.information(self, "Recording", "Recording stopped successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to stop recording: {str(e)}")
                self.video_writer = None
                self.record_button.setText("Start Recording")
            
    def update_statistics(self, detections):
        DIST_THRESHOLD = 80  # Thay đổi threshold thành 80 pixel
        new_counts = []  # Lưu các biển báo vừa tăng count
        for det in detections:
            cls = int(det[-1])
            sign_name = self.model.names[cls]
            x1, y1, x2, y2 = det[:4]
            cx = float(x1 + x2) / 2  # Tọa độ x trung tâm
            cy = float(y1 + y2) / 2  # Tọa độ y trung tâm
            is_new = True
            for (px, py) in self.sign_positions[sign_name]:
                if ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 < DIST_THRESHOLD:
                    is_new = False
                    break
            if is_new:
                self.sign_stats[sign_name] += 1
                self.sign_positions[sign_name].append((cx, cy))
                new_counts.append((sign_name, self.sign_stats[sign_name]))
                
                # Phát âm thanh cảnh báo cho biển báo mới
                sign_name_lower = sign_name.lower()
                sound_file = self.class_to_sound.get(sign_name_lower)
                if sound_file:
                    audio_path = os.path.abspath(os.path.join('sounds_wav', sound_file))
                    print(f"Phát âm thanh cho biển báo: {sign_name} - File: {sound_file}")
                    play_sound_file(audio_path)

        # Update statistics display
        stats_text = "Detected Signs:\n"
        for sign, count in self.sign_stats.items():
            stats_text += f"{sign}: {count}\n"
        self.stats_text.setText(stats_text)
        # Ghi file nếu có biển báo mới
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('sign_statistics.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                writer.writerow(['Time', 'Sign', 'Count'])
            for sign, count in self.sign_stats.items():
                writer.writerow([now, sign, count])
        
    def start_detection(self):
        if self.source_combo.currentText() == "Camera":
            camera_id = int(self.camera_combo.currentText())
            self.cap = cv2.VideoCapture(camera_id)
        else:
            file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", 
                                                     "Video Files (*.mp4 *.avi *.mov)")
            if file_name:
                self.cap = cv2.VideoCapture(file_name)
            else:
                return
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video source")
            return
            
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.source_combo.setEnabled(False)
        self.camera_combo.setEnabled(False)
        
        # Reset statistics
        self.sign_stats.clear()
        self.sign_positions.clear()
        self.stats_text.clear()
        
        # Start timer
        self.timer.start(30)  # 30ms = ~33 FPS
        
    def stop_detection(self):
        # Dừng recording nếu đang ghi
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.record_button.setText("Start Recording")
            
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.source_combo.setEnabled(True)
        self.camera_combo.setEnabled(True)
        self.video_label.clear()
        self.info_label.setText("Video Info: Stopped")
        self.is_paused = False
        self.pause_button.setText("Pause")
        
    def update_frame(self):
        if self.is_paused:
            return
            
        import time
        start_time = time.time()
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return
        # Chỉ xoay nếu nguồn là video file
        if self.source_combo.currentText() == "Video File":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Sử dụng letterbox_fixed thay vì resize thông thường
        img_resized, r, pad_w, pad_h = letterbox_fixed(frame, new_shape=(self.imgsz[0], self.imgsz[1]))
        
        img = img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(img.copy()).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        pred = self.model(img, augment=False)
        conf_thres = float(self.conf_thres.currentText())
        pred = non_max_suppression(pred, conf_thres, 0.45, classes=None, agnostic=False)
        
        for i, det in enumerate(pred):
            if len(det):
                # Scale boxes từ kích thước letterbox về kích thước gốc
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                annotator = Annotator(frame, line_width=2, example=str(self.model.names))
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label_name = self.model.names[c]
                    label = f'{label_name} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
                   
                
                frame = annotator.result()
                self.update_statistics(det)
                
        if self.video_writer is not None:
            self.video_writer.write(frame)
            
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label_size = self.video_label.size()
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            label_size.width(), label_size.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        elapsed = time.time() - start_time
        fps_real = 1.0 / elapsed if elapsed > 0 else 0
        width = w
        height = h
        self.info_label.setText(f"Resolution: {width}x{height} | FPS: {fps_real:.1f}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrafficSignDetector()
    window.show()
    sys.exit(app.exec_())
