# YOLOv8 & DeepSORT Object Tracker

A lightweight, real-time Multi-Object Tracking (MOT) pipeline combining **YOLOv8** for high-accuracy object detection and a custom **SORT (Simple Online and Realtime Tracking)** algorithm using Kalman Filters and Hungarian data association.

## 🚀 Features
- **Real-Time Tracking**: Fast and efficient object tracking optimized for CPU/GPU.
- **Sleek Overlay**: Custom bounding boxes, individual tracker ID labels, and historic path trails.
- **Robust Association**: Keeps tracking objects smoothly even through brief overlaps or obstructions.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Frameworks**: Ultralytics YOLOv8, OpenCV, FilterPy, SciPy, NumPy

## 📦 Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
