"""
Configuration for Behavior Recognition Pipeline
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"

# Input/Output paths
BEHAVIOR_CLIPS_DIR = DATASETS_DIR / "behavior_clips" / "suspicious-actions"
POSE_CACHE_DIR = DATASETS_DIR / "cache" / "pose_sequences"
YOLO_POSE_MODEL = MODELS_DIR / "YOLOv8" / "yolov8m-pose.pt"
LSTM_MODEL_DIR = MODELS_DIR / "lstm_mn"

# YOLO Pose extraction settings
YOLO_CONF_THRESHOLD = 0.3
NUM_KEYPOINTS = 17  # COCO format

# LSTM Training settings
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 7

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Ensure directories exist
POSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LSTM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
