#!/usr/bin/env python3
"""
arcface_simple.py
-----------------
Simple ArcFace face recognition system (matching FaceNet style).
"""

import os
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

# --- Config ---
MODELS_DIR = os.path.join("models", "ArcFace")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "arcface_svm.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "distance_threshold.npy")
BACKBONE_PATH = os.path.join(MODELS_DIR, "resnet50_backbone.pt")

IMAGE_SIZE = 112
EMBEDDING_SIZE = 512

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Same backbone as training ---
class ArcFaceBackbone(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Linear(2048, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = self.bn(x)
        return nn.functional.normalize(x, p=2, dim=1)

class ArcFaceSimple:
    def __init__(self):
        self.embedder = None
        self.classifier = None
        self.encoder = None
        self.threshold = None
        self.loaded = False
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_models(self):
        """Load all trained models."""
        try:
            # Load backbone
            self.embedder = ArcFaceBackbone(EMBEDDING_SIZE).to(device)
            self.embedder.load_state_dict(torch.load(BACKBONE_PATH, map_location=device))
            self.embedder.eval()
            
            # Load classifier and encoder
            self.classifier = joblib.load(CLASSIFIER_PATH)
            self.encoder = joblib.load(ENCODER_PATH)
            self.threshold = np.load(THRESHOLD_PATH)
            
            self.loaded = True
            print(f"[INFO] ArcFace models loaded successfully!")
            print(f"   Classes: {', '.join(self.encoder.classes_)}")
            print(f"   Threshold: {self.threshold:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            self.loaded = False
    
    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Get embedding for face image."""
        if not self.loaded:
            return None
        
        # Resize and preprocess
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        tensor = self.transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = self.embedder(tensor).cpu().numpy().flatten()
        return embedding
    
    def recognize_face(self, face_img: np.ndarray) -> tuple:
        """Recognize face and return (name, confidence)."""
        if not self.loaded:
            return "Unknown", 0.0
        
        try:
            # Get embedding
            embedding = self.get_embedding(face_img)
            if embedding is None:
                return "Unknown", 0.0
            
            # Get probabilities
            probs = self.classifier.predict_proba([embedding])[0]
            max_prob_idx = np.argmax(probs)
            max_prob = probs[max_prob_idx]
            
            # Get predicted class
            predicted_class = self.classifier.classes_[max_prob_idx]
            name = self.encoder.classes_[predicted_class]
            
            # Check against threshold (convert probability to distance-like score)
            confidence = max_prob
            if confidence < 0.5:  # Adjust this threshold as needed
                return "Unknown", confidence
            
            return name, confidence
            
        except Exception as e:
            print(f"[ERROR] Recognition failed: {e}")
            return "Unknown", 0.0

# Global instance (like FaceNet)
arcface_recognizer = ArcFaceSimple()

def initialize_arcface():
    """Initialize ArcFace system."""
    arcface_recognizer.load_models()
    return arcface_recognizer.loaded

def recognize_face(face_img: np.ndarray) -> tuple:
    """Simple face recognition function (like FaceNet)."""
    return arcface_recognizer.recognize_face(face_img)