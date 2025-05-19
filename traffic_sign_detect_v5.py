import sys
import cv2
import torch
import time
import pyttsx3
from datetime import datetime
from collections import defaultdict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QComboBox, 
                           QFileDialog, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import sys
import os
import csv

# Create recordings directory if it doesn't exist
os.makedirs('recordings', exist_ok=True)

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.path.dirname(sys.modules["PyQt5"].__file__), "Qt5", "plugins")
sys.path.append('yolov5')  # Add yolov5 directory to Python path
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device

class TrafficSignDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Sign Detection")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Tốc độ nói
        self.engine.setProperty('volume', 1.0)  # Âm lượng
        
        # Initialize YOLOv5 model
        self.device = select_device('')
        self.model = DetectMultiBackend('weights/best_2.pt', device=self.device)
        self.stride = self.model.stride
        self.imgsz = check_img_size((640, 640), s=self.stride)
        
        # Initialize video capture and recording
        self.cap = None
        self.video_writer = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Initialize statistics
        self.sign_stats = defaultdict(int)
        self.sign_positions = defaultdict(list)  # Lưu các vị trí đã đếm cho từng loại biển báo
        self.important_signs = {
            'cam re trai': 'CẤM RẼ TRÁI',
            'cam re phai': 'CẤM RẼ PHẢI',
            'cam do xe': 'CẤM ĐỖ XE',
            'cam dung xe va do xe': 'CẤM DỪNG/ĐỖ XE',
            'cam oto re trai': 'CẤM Ô TÔ RẼ TRÁI',
            'cam bam coi': 'CẤM BẤM CÒI',
            'cam xe may 2 banh': 'CẤM XE MÁY 2 BÁNH',
            'cam di nguoc chieu': 'CẤM ĐI NGƯỢC CHIỀU',
            'toc do toi da 40 km/h': 'TỐC ĐỘ TỐI ĐA 40 KM/H',
            'toc do toi da 60 km/h': 'TỐC ĐỘ TỐI ĐA 60 KM/H',
            'toc do toi da 80 km/h': 'TỐC ĐỘ TỐI ĐA 80 KM/H',
            'cam quay dau': 'CẤM QUAY ĐẦU',
            'tre em qua duong': 'TRẺ EM QUA ĐƯỜNG',
            'duong giao nhau': 'ĐƯỜNG GIAO NHAU',
            'giao nhau voi duong uu tien': 'GIAO NHAU VỚI ĐƯỜNG ƯU TIÊN',
            'giao nhau voi duong khong uu tien': 'GIAO NHAU VỚI ĐƯỜNG KHÔNG ƯU TIÊN',
            'noi giao nhau theo vong xuyen': 'GIAO NHAU VÒNG XUYÊN',
            'background': None
        }
        
        # Initialize UI
        self.init_ui()
        
    def init_ui(self):
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
        # Buttons
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_detection)
        right_layout.addWidget(self.start_button)
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
        self.stats_text.setMaximumHeight(150)
        right_layout.addWidget(self.stats_text)
        # Alerts display
        self.alerts_label = QLabel("Important Sign Alerts:")
        self.alerts_label.setStyleSheet('font-size: 17px; color: #ff5555; font-weight: bold;')
        right_layout.addWidget(self.alerts_label)
        self.alerts_text = QTextEdit()
        self.alerts_text.setReadOnly(True)
        self.alerts_text.setMaximumHeight(150)
        right_layout.addWidget(self.alerts_text)
        # Add panels to main layout
        layout.addWidget(left_panel, stretch=2)
        layout.addWidget(right_panel, stretch=1)
        
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
                
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Quay lại đầu video nếu là file
                
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
        DIST_THRESHOLD = 50  # pixel
        new_counts = []  # Lưu các biển báo vừa tăng count
        for det in detections:
            cls = int(det[-1])
            sign_name = self.model.names[cls]
            x1, y1, x2, y2 = det[:4]
            cx = float(x1 + x2) / 2
            cy = float(y1 + y2) / 2
            is_new = True
            for (px, py) in self.sign_positions[sign_name]:
                if ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 < DIST_THRESHOLD:
                    is_new = False
                    break
            if is_new:
                self.sign_stats[sign_name] += 1
                self.sign_positions[sign_name].append((cx, cy))
                new_counts.append((sign_name, self.sign_stats[sign_name]))
        # Update statistics display
        stats_text = "Detected Signs:\n"
        for sign, count in self.sign_stats.items():
            stats_text += f"{sign}: {count}\n"
        self.stats_text.setText(stats_text)
        # Chỉ ghi vào file khi count tăng
        if new_counts:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open('sign_statistics.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if csvfile.tell() == 0:
                    writer.writerow(['Time', 'Sign', 'Count'])
                for sign, count in new_counts:
                    writer.writerow([now, sign, count])
        # Check for important signs
        alerts = []
        for det in detections:
            cls = int(det[-1])
            sign_name = self.model.names[cls].lower()
            conf = float(det[-2])
            for important_sign in self.important_signs:
                if important_sign in sign_name and conf > 0.5:
                    alert = f"ALERT: {self.important_signs[important_sign]} detected! (Confidence: {conf:.2f})"
                    if alert not in alerts:
                        alerts.append(alert)
                        # Thông báo qua loa
                        try:
                            self.engine.say(f"Chú ý! {self.important_signs[important_sign]}")
                            self.engine.runAndWait()
                        except Exception as e:
                            print(f"Error in text-to-speech: {str(e)}")
        if alerts:
            self.alerts_text.setText("\n".join(alerts))
        
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
        self.record_button.setEnabled(True)
        self.source_combo.setEnabled(False)
        self.camera_combo.setEnabled(False)
        
        # Reset statistics
        self.sign_stats.clear()
        self.sign_positions.clear()
        self.stats_text.clear()
        self.alerts_text.clear()
        
        # Start timer
        self.timer.start(30)  # 30ms = ~33 FPS
        
    def stop_detection(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
            
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.record_button.setText("Start Recording")
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.source_combo.setEnabled(True)
        self.camera_combo.setEnabled(True)
        self.video_label.clear()
        self.info_label.setText("Video Info: Stopped")
        
    def update_frame(self):
        import time
        start_time = time.time()
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return
        # Chỉ xoay nếu nguồn là video file
        if self.source_combo.currentText() == "Video File":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # BỎ QUA bước nhận diện model để kiểm tra FPS gốc
        img = cv2.resize(frame, (self.imgsz[0], self.imgsz[1]))
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(img.copy()).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=False)
        conf_thres = float(self.conf_thres.currentText())
        pred = non_max_suppression(pred, conf_thres, 0.45, classes=None, agnostic=False)
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                annotator = Annotator(frame, line_width=2, example=str(self.model.names))
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{self.model.names[c]} {conf:.2f}'
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