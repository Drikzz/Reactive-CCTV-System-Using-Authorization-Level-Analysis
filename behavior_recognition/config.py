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
BEHAVIOR_CLIPS_DIR = DATASETS_DIR / "behavior_clips"  # Base directory containing both datasets
BEHAVIOR_DATASETS = ["suspicious-actions", "neutral-actions"]  # Datasets to load
POSE_CACHE_DIR = DATASETS_DIR / "cache" / "pose_sequences"
YOLO_POSE_MODEL = MODELS_DIR / "YOLOv8" / "yolov8m-pose.pt"
LSTM_MODEL_DIR = MODELS_DIR / "lstm_mn"

# YOLO Pose extraction settings
YOLO_CONF_THRESHOLD = 0.25  # Lowered from 0.75 - too high was causing detection failures
NUM_KEYPOINTS = 17  # COCO format
KEYPOINT_FEATURES = 4  # [x, y, dx, dy] - position + velocity
FIXED_SEQUENCE_LENGTH = None  # Sample all videos to this many frames (set to None to disable) 90
MULTI_PERSON_STRATEGY = 'largest'  # 'largest' = closest to camera, 'first' = first detection

# LSTM Training settings
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
HIDDEN_DIM = 256 #changed from 128
NUM_LAYERS = 2
DROPOUT = 0.4 #changed from 0.3
EARLY_STOPPING_PATIENCE = 10 #changed from 7

# Data split ratios
TRAIN_RATIO = 0.7 #changed from 0.8
VAL_RATIO = 0.15 #changed from 0.1
TEST_RATIO = 0.15 #changed from 0.1

# Ensure directories exist
POSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LSTM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
