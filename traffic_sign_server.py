import os
import sys
import cv2
import torch
import numpy as np
import socket
import struct
import pickle

# Thêm yolov5 vào đường dẫn Python
sys.path.append('yolov5')
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

def letterbox_fixed(img, new_shape=(320, 320), color=(114, 114, 114)):
    """
    Thay đổi kích thước ảnh và thêm padding để giữ tỷ lệ khung hình.
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

class TrafficSignServer:
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        
        # Khởi tạo model YOLOv5
        self.device = select_device('')
        self.model = DetectMultiBackend('weights/best_2.pt', device=self.device)
        self.stride = self.model.stride
        self.imgsz = check_img_size((640, 640), s=self.stride)
        
        # Khởi tạo socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server listening on {self.host}:{self.port}")
        
    def process_frame(self, frame):
        # Xử lý frame với YOLO
        img_resized, r, pad_w, pad_h = letterbox_fixed(frame, new_shape=(self.imgsz[0], self.imgsz[1]))
        img = img_resized.transpose((2, 0, 1))[::-1]
        img = torch.from_numpy(img.copy()).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Thực hiện detection
        pred = self.model(img, augment=False)
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
        
        # Chuyển đổi kết quả thành định dạng có thể gửi
        detections = []
        for i, det in enumerate(pred):
            if len(det):
                # Scale boxes về frame gốc
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    class_id = int(cls)
                    class_name = self.model.names[class_id]
                    detections.append({
                        'class': class_id,
                        'class_name': class_name,
                        'confidence': float(conf),
                        'box': [float(x) for x in xyxy]
                    })
        
        return detections
        
    def handle_client(self, client_socket):
        try:
            while True:
                # Nhận kích thước frame
                data = client_socket.recv(4)
                if not data:
                    break
                frame_size = struct.unpack(">L", data)[0]
                
                # Nhận frame
                frame_data = b""
                while len(frame_data) < frame_size:
                    packet = client_socket.recv(frame_size - len(frame_data))
                    if not packet:
                        break
                    frame_data += packet
                
                if len(frame_data) != frame_size:
                    break
                    
                # Giải nén frame
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                # Xử lý frame
                detections = self.process_frame(frame)
                
                # Gửi số lượng detections
                client_socket.sendall(struct.pack(">L", len(detections)))
                
                # Gửi thông tin detections
                for det in detections:
                    det_bytes = pickle.dumps(det)
                    client_socket.sendall(struct.pack(">L", len(det_bytes)))
                    client_socket.sendall(det_bytes)
                    
        except Exception as e:
            print(f"Error handling client: {str(e)}")
        finally:
            client_socket.close()
            
    def run(self):
        try:
            while True:
                print("Waiting for client connection...")
                client_socket, address = self.server_socket.accept()
                print(f"Connected to client: {address}")
                self.handle_client(client_socket)
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.server_socket.close()

if __name__ == '__main__':
    server = TrafficSignServer()
    server.run() 
