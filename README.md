Camera–LiDAR perception demo using KITTI dataset.
Includes YOLOv8 2D detection, monocular depth approximation (MiDaS),
and KITTI ground-truth 3D bounding boxes visualized in both image
and LiDAR space for autonomous robotics research.

📁 Structure
kitti-yolov8-3d-approx-perception/
│
├── README.md
├── requirements.txt
├── kitti_3d_bbox_demo.py
│
├── assets/
│   ├── demo_1.png
│   ├── demo_2.png
│   ├── demo_3.png
│
└── .gitignore

 
# KITTI Perception Demo: YOLOv8 + Depth + 3D Bounding Boxes

This repository demonstrates a **robotics-grade perception pipeline**
using the **KITTI dataset**, combining:

- 🔍 **YOLOv8** for 2D object detection
- 🌊 **MiDaS** for monocular depth estimation (3D approximation)
- 📦 **KITTI ground-truth 3D bounding boxes**
- ☁️ **LiDAR point cloud visualization** (Open3D)

The goal is to visually and technically demonstrate how **real robots
perceive static and dynamic objects in 3D**.

---

## 🚀 Features

✅ 2D Object Detection (YOLOv8)  
✅ Depth Estimation per Object (MiDaS – monocular)  
✅ KITTI Ground-Truth 3D Bounding Boxes (Camera & LiDAR frames)  
✅ Static vs Dynamic Object Classification  
✅ Side-by-side Image + LiDAR Visualization  
✅ Frame-by-frame playback with pause & step controls  

---

## 🧠 Pipeline Overview



Camera Image ──▶ YOLOv8 ──▶ 2D Boxes
│
├──▶ MiDaS ──▶ Relative Depth
│
LiDAR Point Cloud ──▶ KITTI GT ──▶ True 3D Boxes


This allows:
- **Fast demos on CPU**
- **Validation against true 3D geometry**
- **Scalability to PointPillars / BEV / GPU models**
---

## 📸 Demo Results

### Camera + 3D Boxes
<img width="1257" height="463" alt="Screenshot From 2026-02-03 19-05-42" src="https://github.com/user-attachments/assets/3eb93325-3e96-4057-bd95-64a1ab660da7" />



### Multi-object Detection
<img width="1257" height="463" alt="Screenshot From 2026-02-03 19-06-04" src="https://github.com/user-attachments/assets/f0821963-8272-4e48-9824-7731a027e3b0" />


### Urban Scenario
<img width="1257" height="463" alt="Screenshot From 2026-02-03 19-06-10" src="https://github.com/user-attachments/assets/507cb8bc-35c7-43c8-83c3-9307f55e1b4a" />
<img width="1257" height="463" alt="Screenshot From 2026-02-03 19-06-20" src="https://github.com/user-attachments/assets/294cca85-720c-49b8-a6d2-475ec23258cc" />


---

## 🛠 Installation

### 1️⃣ Create Conda Environment
```bash
conda create -n perception python=3.11 -y
conda activate perception

2️⃣ Install Dependencies
pip install torch torchvision torchaudio
pip install ultralytics opencv-python matplotlib open3d timm


📂 Dataset Setup

Download KITTI:

data_object_image_2

data_object_velodyne

data_object_calib

data_object_label_2

Folder structure:

kitti/
├── data_object_image_2/training/image_2
├── data_object_velodyne/training/velodyne
├── data_object_calib/training/calib
├── data_object_label_2/training/label_2


Update path in code:

KITTI_ROOT = "/path/to/kitti"
