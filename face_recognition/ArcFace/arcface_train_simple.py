#!/usr/bin/env python3
"""
arcface_train_simple.py
-----------------------
Train an SVM classifier on face embeddings extracted with ArcFace backbone.

Pipeline (matching FaceNet):
    - Load aligned faces from datasets/faces/<person>/
    - Apply augmentation to training images
    - Compute embeddings with ResNet50 backbone
    - Encode labels
    - Train linear SVM
    - Compute distance threshold (mean intra-class + 2σ)
    - Save models, encoder, and threshold

Output:
    models/ArcFace/
        arcface_svm.joblib
        label_encoder.joblib
        distance_threshold.npy
        resnet50_backbone.pt
"""

import os
import cv2
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

# --- Config ---
DATASET_DIR = "datasets/faces"
MODELS_DIR = os.path.join("models", "ArcFace")

CLASSIFIER_PATH = os.path.join(MODELS_DIR, "arcface_svm.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "distance_threshold.npy")
BACKBONE_PATH = os.path.join(MODELS_DIR, "resnet50_backbone.pt")

IMAGE_SIZE = 112  # ArcFace standard
EMBEDDING_SIZE = 512
TEST_SPLIT = 0.2
RANDOM_STATE = 42

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Simple ResNet50 Backbone (like ArcFace) ---
class ArcFaceBackbone(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Add embedding layer
        self.embedding = nn.Linear(2048, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.embedding(x)
        x = self.bn(x)
        return nn.functional.normalize(x, p=2, dim=1)  # L2 normalize

# Initialize backbone
embedder = ArcFaceBackbone(embedding_size=EMBEDDING_SIZE).eval().to(device)

# --- Augmentation pipeline (same as FaceNet) ---
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_embedding(img_bgr: np.ndarray, augment: bool = False) -> np.ndarray:
    """Resize, normalize, augment (if train), and embed image."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    if augment:
        tensor = train_transform(img)
    else:
        tensor = test_transform(img)
    
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = embedder(tensor).cpu().numpy().flatten()
    return emb

def load_dataset(dataset_dir: str):
    """Load dataset images and labels from folders (same as FaceNet)."""
    X, y = [], []
    for person in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for file in os.listdir(person_dir):
            path = os.path.join(person_dir, file)
            img = cv2.imread(path)
            if img is None:
                continue
            X.append(img)
            y.append(person)
    return X, y

def compute_threshold(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute intra-class distance threshold (same as FaceNet)."""
    distances = []
    for label in np.unique(labels):
        idxs = np.where(labels == label)[0]
        if len(idxs) < 2:
            continue
        embs = embeddings[idxs]
        center = np.mean(embs, axis=0)
        dists = np.linalg.norm(embs - center, axis=1)
        distances.extend(dists)
    return float(np.mean(distances) + 2 * np.std(distances))

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Load dataset ---
    print("[INFO] Loading dataset...")
    X, y = load_dataset(DATASET_DIR)
    print(f"[INFO] Loaded {len(X)} images from {len(set(y))} classes.")

    # --- Split ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_STATE
        )
    except ValueError:
        X_train, X_test, y_train, y_test = X, [], y, []
        print("[WARN] Stratified split failed (small dataset). Using all data for training.")

    # --- Encode labels ---
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test) if len(y_test) > 0 else []

    # --- Embeddings ---
    print("[INFO] Computing embeddings...")
    X_train_emb = np.array([get_embedding(img, augment=True) for img in tqdm(X_train, desc="Train")])

    X_test_emb = None
    if len(X_test) > 0:
        X_test_emb = np.array([get_embedding(img, augment=False) for img in tqdm(X_test, desc="Test")])

    # --- Train classifier ---
    print("[INFO] Training SVM classifier...")
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train_emb, y_train_enc)

    # --- Evaluate ---
    if X_test_emb is not None and len(X_test_emb) > 0:
        y_pred = clf.predict(X_test_emb)
        acc = accuracy_score(y_test_enc, y_pred)
        print(f"[RESULT] Test Accuracy: {acc:.4f}")
    else:
        print("[WARN] No test set; accuracy not computed.")

    # --- Threshold ---
    print("[INFO] Calculating distance threshold...")
    threshold = compute_threshold(X_train_emb, y_train_enc)
    print(f"[INFO] Distance threshold set to {threshold:.4f}")

    # --- Save ---
    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    np.save(THRESHOLD_PATH, threshold)
    torch.save(embedder.state_dict(), BACKBONE_PATH)

    print(f"[DONE] Models saved to {MODELS_DIR}/")

if __name__ == "__main__":
    main()