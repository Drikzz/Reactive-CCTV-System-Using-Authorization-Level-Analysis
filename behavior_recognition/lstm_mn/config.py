"""
Configuration for MobileNetV2 + LSTM Behavior Recognition Pipeline
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"

# Input/Output paths
BEHAVIOR_CLIPS_DIR = DATASETS_DIR / "behavior_clips"  # Base directory containing video datasets
BEHAVIOR_DATASETS = ["suspicious-actions", "neutral-actions"]  # Datasets to load
BEHAVIOR_FRAMES_DIR = DATASETS_DIR / "behavior_frames"  # Extracted frames with person tracking
LSTM_MODEL_DIR = MODELS_DIR / "lstm_mn"  # Model save directory

# YOLO Detection Model
YOLO_DETECTION_MODEL = MODELS_DIR / "YOLOv8" / "yolov8m.pt"  # For person detection

# Video Processing Settings
TARGET_FPS = 30  # Extract frames at 30fps
FRAME_WIDTH = 224  # MobileNetV2 input size
FRAME_HEIGHT = 224  # MobileNetV2 input size
SEQUENCE_LENGTH = 30  # Number of frames per sequence (1 second at 30fps)
YOLO_CONF_THRESHOLD = 0.3  # Confidence threshold for person detection
YOLO_IOU_THRESHOLD = 0.5  # IOU threshold for tracking
MIN_DETECTION_CONFIDENCE = 0.25  # Minimum confidence to consider a detection valid

# Model Architecture Settings
MOBILENET_FEATURE_DIM = 1280  # MobileNetV2 output feature dimension
LSTM_HIDDEN_DIM = 256  # LSTM hidden state dimension
LSTM_NUM_LAYERS = 2  # Number of LSTM layers
LSTM_DROPOUT = 0.3  # Dropout rate for LSTM

# Training Settings
BATCH_SIZE = 8  # Smaller batch size due to sequence processing
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Data Split Ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Person Tracking Settings
ENABLE_PERSON_TRACKING = True  # Set to False to use full frames
MAX_PERSONS_PER_VIDEO = 2  # Maximum number of people to track per video
MIN_TRACK_LENGTH = 15  # Minimum number of frames to consider a valid track

# Normalization Settings (ImageNet statistics for MobileNetV2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Ensure directories exist
BEHAVIOR_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
LSTM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
