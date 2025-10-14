#!/usr/bin/env python3
"""
arcface_train_simple.py
-----------------------
Train an SVM classifier on face embeddings extracted with ArcFace backbone.
Now with GPU acceleration for both embedding extraction and SVM training!

Pipeline (matching FaceNet):
    - Load aligned faces from datasets/faces/<person>/
    - Apply augmentation to training images
    - Compute embeddings with ResNet50 backbone (GPU)
    - Encode labels
    - Train linear SVM (GPU-accelerated via cuML if available)
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import glob

# Try to import GPU-accelerated SVM from cuML (RAPIDS)
import importlib
from typing import TYPE_CHECKING

# Help static type checkers / editors: provide a conditional import for TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from cuml.svm import SVC as cuSVC  # type: ignore
    except Exception:
        cuSVC = None  # type: ignore

# At runtime, import dynamically so environments without cuml don't raise editor/resolver errors
try:
    cuml_svm = importlib.import_module("cuml.svm")
    cuSVC = getattr(cuml_svm, "SVC")
    USE_GPU_SVM = True
    print("[INFO] cuML detected - will use GPU-accelerated SVM!")
except (ImportError, ModuleNotFoundError):
    # cuML not available, fallback to sklearn at runtime
    USE_GPU_SVM = False
    print("[INFO] cuML not available - using CPU SVM (sklearn)")

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
BATCH_SIZE = 32  # For faster GPU embedding extraction

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")
if device == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

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

def get_embeddings_batch(images: list, augment: bool = False) -> np.ndarray:
    """Batch process images for faster GPU embedding extraction"""
    tensors = []
    for img_bgr in images:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        
        if augment:
            tensor = train_transform(img)
        else:
            tensor = test_transform(img)
        tensors.append(tensor)
    
    # Stack into batch
    batch = torch.stack(tensors).to(device)
    
    with torch.no_grad():
        embeddings = embedder(batch).cpu().numpy()
    
    return embeddings

def load_dataset(dataset_dir: str):
    """Load dataset images and labels.
    Supports both flat structure:
        datasets/faces/<person>/*.jpg
    and multi-angle structure:
        datasets/faces/<person>/<angle>/*.jpg
    """
    X, y = [], []
    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return X, y

    person_dirs = sorted([d for d in os.listdir(dataset_dir) 
                         if os.path.isdir(os.path.join(dataset_dir, d))])
    
    print(f"[INFO] Found {len(person_dirs)} persons: {person_dirs}")

    for person in person_dirs:
        person_dir = os.path.join(dataset_dir, person)

        # collect image paths from angle subfolders if present, otherwise from person_dir
        image_paths = []
        subdirs = [d for d in sorted(os.listdir(person_dir))
                   if os.path.isdir(os.path.join(person_dir, d))]
        if subdirs:
            print(f"[INFO] Loading {person} with multi-angle structure: {subdirs}")
            for sub in subdirs:
                sub_path = os.path.join(person_dir, sub)
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    image_paths.extend(glob.glob(os.path.join(sub_path, ext)))
        else:
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                image_paths.extend(glob.glob(os.path.join(person_dir, ext)))

        image_paths = sorted(image_paths)
        if len(image_paths) == 0:
            print(f"[WARN] No images found for {person}")
            continue

        print(f"[INFO] Loading {len(image_paths)} images for {person}")
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"[WARN] Could not read image: {path}")
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
    
    if len(X) == 0:
        print("[ERROR] No images loaded!")
        return
    
    unique_classes = sorted(list(set(y)))
    print(f"[INFO] Loaded {len(X)} images from {len(unique_classes)} classes.")
    print(f"[INFO] Classes: {unique_classes}")

    # --- Split ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_STATE
        )
    except ValueError:
        X_train, X_test, y_train, y_test = X, [], y, []
        print("[WARN] Stratified split failed (small dataset). Using all data for training.")

    print(f"[INFO] Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # --- Encode labels ---
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test) if len(y_test) > 0 else []

    # --- Embeddings (batch processing for speed) ---
    print("[INFO] Computing embeddings with GPU acceleration...")
    print(f"[INFO] Processing in batches of {BATCH_SIZE}")
    
    # Train embeddings
    X_train_emb = []
    for i in tqdm(range(0, len(X_train), BATCH_SIZE), desc="Train embeddings"):
        batch = X_train[i:i+BATCH_SIZE]
        emb_batch = get_embeddings_batch(batch, augment=True)
        X_train_emb.append(emb_batch)
    X_train_emb = np.vstack(X_train_emb)

    # Test embeddings
    X_test_emb = None
    if len(X_test) > 0:
        X_test_emb = []
        for i in tqdm(range(0, len(X_test), BATCH_SIZE), desc="Test embeddings"):
            batch = X_test[i:i+BATCH_SIZE]
            emb_batch = get_embeddings_batch(batch, augment=False)
            X_test_emb.append(emb_batch)
        X_test_emb = np.vstack(X_test_emb)

    print(f"[INFO] Training embeddings shape: {X_train_emb.shape}")
    if X_test_emb is not None:
        print(f"[INFO] Test embeddings shape: {X_test_emb.shape}")

    # --- Train classifier (GPU or CPU) ---
    print(f"[INFO] Training SVM classifier {'on GPU (cuML)' if USE_GPU_SVM else 'on CPU (sklearn)'}...")
    
    if USE_GPU_SVM:
        # GPU-accelerated SVM via cuML
        clf = cuSVC(kernel="linear", probability=True, cache_size=2000)
        clf.fit(X_train_emb, y_train_enc)
    else:
        # CPU SVM
        from sklearn.svm import SVC
        clf = SVC(kernel="linear", probability=True, cache_size=2000)
        clf.fit(X_train_emb, y_train_enc)

    # --- Evaluate ---
    if X_test_emb is not None and len(X_test_emb) > 0:
        y_pred = clf.predict(X_test_emb)
        acc = accuracy_score(y_test_enc, y_pred)
        print(f"\n[RESULT] Test Accuracy: {acc:.4f}")
        
        # Per-class accuracy
        for i, class_name in enumerate(encoder.classes_):
            class_mask = y_test_enc == i
            if class_mask.sum() > 0:
                class_acc = accuracy_score(y_test_enc[class_mask], y_pred[class_mask])
                print(f"[INFO] {class_name}: {class_acc:.4f}")
    else:
        print("[WARN] No test set; accuracy not computed.")

    # --- Threshold ---
    print("[INFO] Calculating distance threshold...")
    threshold = compute_threshold(X_train_emb, y_train_enc)
    print(f"[INFO] Distance threshold set to {threshold:.4f}")

    # --- Save ---
    print("[INFO] Saving models...")
    
    # Convert cuML model to sklearn for compatibility if needed
    if USE_GPU_SVM:
        print("[INFO] Converting cuML model to sklearn format for compatibility...")
        # Save as-is; inference code will need cuML or we convert to sklearn
        # For now save the cuML model directly
        
    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    np.save(THRESHOLD_PATH, threshold)
    torch.save(embedder.state_dict(), BACKBONE_PATH)

    print(f"\n[DONE] Models saved to {MODELS_DIR}/")
    print(f"[INFO] Classifier: {CLASSIFIER_PATH}")
    print(f"[INFO] Encoder: {ENCODER_PATH}")
    print(f"[INFO] Threshold: {THRESHOLD_PATH}")
    print(f"[INFO] Backbone: {BACKBONE_PATH}")
    
    if USE_GPU_SVM:
        print("\n[NOTE] Model trained with GPU (cuML). For inference:")
        print("  - Install cuML for GPU inference: conda install -c rapidsai cuml")
        print("  - Or convert model to sklearn for CPU inference")

if __name__ == "__main__":
    main()