# Thesis: Real-Time Face Recognition & Behavior Analysis using YOLOv8

## 🔧 Setup
1. Use Google Colab notebooks in each folder
2. Mount your Google Drive to access datasets and models
3. Ensure folder paths are consistent across notebooks

## 📁 Folder Guide
- `yolov8_setup/`: Basic YOLOv8 install + test
- `face_recognition/`: Face detection & recognition pipeline
- `behavior_analysis/`: Movement analysis and behavior flagging
- `utils/`: Python helper scripts (e.g., drawing, encoding, matching)
- `datasets/`: Your image, video, and known face datasets
- `models/`: Custom trained YOLO models

## 🚀 How to Run
- Run `yolov8_setup.ipynb` to install dependencies
- Use `face_recognition.ipynb` to test face recognition
- Use `behavior_analysis.ipynb` to test activity detection

## 📌 Notes
- Colab needs runtime set to GPU (T4 recommended)
- Use `cv2_imshow()` instead of `imshow()` in notebooks
- Use `ArcFace` or `FaceNet` for face identity matching
