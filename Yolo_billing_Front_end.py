# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:14:54 2024

@author: HP
"""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2
import torch
from ultralytics import YOLO  # For YOLOv8 models

class YOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Prediction App")
        self.setGeometry(100, 100, 800, 600)

        # YOLO model
        self.model = None

        # Create UI elements
        self.video_label = QLabel("Video Feed")
        self.video_label.setScaledContents(True)
        self.load_model_button = QPushButton("Load YOLO Model")
        #self.load_video_button = QPushButton("Load Video")
        self.load_image_button = QPushButton("Load Image")
        self.detected_classes = QTextEdit()
        self.detected_classes.setReadOnly(True)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.load_model_button)
        #layout.addWidget(self.load_video_button)
        layout.addWidget(self.load_image_button)
        layout.addWidget(self.video_label)
        layout.addWidget(QLabel("Detected Classes:"))
        layout.addWidget(self.detected_classes)

        # Container widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connect buttons to methods
        self.load_model_button.clicked.connect(self.load_model)
        #self.load_video_button.clicked.connect(self.load_video)
        self.load_image_button.clicked.connect(self.load_image)

        # Video variables
        self.cap = None
        self.timer = QTimer()

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "D:/Computer Vision/model", "YOLO Model Files (*.pt)")
        if model_path:
            try:
                self.model = YOLO(model_path)
                self.detected_classes.append(f"Model loaded successfully: {os.path.basename(model_path)}")
            except Exception as e:
                self.detected_classes.append(f"Error loading model: {e}")

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.timer.timeout.connect(self.process_video_frame)
            self.timer.start(30)  # Process frames at ~30 FPS

    def process_video_frame(self):
        if self.cap is None or not self.cap.isOpened() or self.model is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        # Resize the frame to 1/3 of its original size
        height, width = frame.shape[:2]
        new_width = int(width / 5)
        new_height = int(height / 5)
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # YOLO prediction
        results = self.model.predict(resized_frame, imgsz=320, conf=0.5)
        annotated_frame = results[0].plot()

        # Convert frame to QImage
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Update video label
        self.video_label.setPixmap(pixmap)

        # Update detected classes
        detected = results[0].names
        self.detected_classes.clear()
        for class_id, class_name in detected.items():
            self.detected_classes.append(f"Class ID: {class_id}, Class Name: {class_name}")

    def load_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "D:/Computer Vision/Pencil_dataset/test/images", "Image Files (*.jpg *.png)")
        if image_path and self.model is not None:
            frame = cv2.imread(image_path)
    
            # Resize the image to 1/5 of its original size
            height, width = frame.shape[:2]
            new_width = int(width / 5)
            new_height = int(height / 5)
            resized_frame = cv2.resize(frame, (new_width, new_height))
    
            # Run YOLOv8 prediction
            results = self.model.predict(resized_frame, imgsz=320, conf=0.5)
            result = results[0]  # First image result
            boxes = result.boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
    
            # Get names from model
            class_names = self.model.names
    
            # Items and prices
            item_prices = {
                'Pencil': 10,
                'Eraser':6,
                'Sharpener': 5,
                'Ruler': 15
            }
    
            item_counts = {
                'Pencil': 0,
                'Eraser':0,
                'Sharpener': 0,
                'Ruler': 0
            }
    
            for cls_id in class_ids:
                name = class_names[cls_id]
                if name in item_counts:
                    item_counts[name] += 1
    
            # Draw annotated results on image
            annotated_frame = result.plot()
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap)
    
            # Show results in text area
            self.detected_classes.clear()
            self.detected_classes.append("🧾 Detected Items:")
            total_price = 0
            for item, count in item_counts.items():
                price = item_prices[item] * count
                total_price += price
                self.detected_classes.append(f"{item}: {count} x {item_prices[item]} = {price} PKR")
    
            self.detected_classes.append(f"\nTotal Bill: {total_price} PKR")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())
