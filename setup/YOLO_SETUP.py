# yolov8_setup.py

import os
import cv2
from ultralytics import YOLO

def install_requirements():
    os.system('pip install ultralytics opencv-python')

def load_model(model_path="yolov8n.pt"):
    # Load YOLOv8 model
    model = YOLO(model_path)
    return model

def run_inference(model, image_path):
    # Run YOLOv8 model on an image
    results = model(image_path)
    return results

def main():
    # Install dependencies
    install_requirements()
    
    # Load YOLOv8 model
    model = load_model()
    
    # Run inference on a test image
    image_path = "sample_image.jpg"  # Update with the path to your image
    results = run_inference(model, image_path)
    
    # Show the results
    results.show()

if __name__ == "__main__":
    main()
