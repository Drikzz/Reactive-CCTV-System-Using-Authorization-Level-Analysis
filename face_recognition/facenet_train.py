#!/usr/bin/env python3
"""
facenet_train.py
----------------
Train an SVM classifier on face embeddings extracted with facenet-pytorch.

Pipeline:
    - Load aligned faces from datasets/faces/<person>/
    - Apply augmentation to training images
    - Compute embeddings with InceptionResnetV1 (facenet-pytorch)
    - Encode labels
    - Train linear SVM
    - Compute distance threshold (mean intra-class + 20)
    - Save models, encoder, and threshold

Output:
    models/
        facenet_svm.joblib
        label_encoder.joblib
        distance_threshold.npy
        inception_resnet_v1.pt
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
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# --- Config ---
DATASET_DIR = "datasets/faces"
MODELS_DIR = MODELS_DIR = os.path.join("models", "FaceNet")

CLASSIFIER_PATH = os.path.join(MODELS_DIR, "facenet_svm.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "distance_threshold.npy")
EMBEDDER_PATH = os.path.join(MODELS_DIR, "inception_resnet_v1.pt")

IMAGE_SIZE = 160
TEST_SPLIT = 0.2
RANDOM_STATE = 42

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# --- Augmentation pipeline ---
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])


def get_embedding(img_bgr: np.ndarray, augment: bool = False) -> np.ndarray:
    """Resize, normalize, augment (if train), and embed image."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    if augment:
        # Apply augmentation with torchvision
        img = train_transform(img)
        img = img.permute(1, 2, 0).numpy() * 255
        img = img.astype(np.uint8)

    tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 127.5) / 128.0  # normalize [-1, 1]
    tensor = tensor.to(device)

    with torch.no_grad():
        emb = embedder(tensor).cpu().numpy().flatten()
    return emb


def load_dataset(dataset_dir: str):
    """Load dataset images and labels from folders."""
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
    return X, y  # return lists, not numpy arrays


def compute_threshold(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute intra-class distance threshold (mean + 2σ)."""
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
    torch.save(embedder.state_dict(), EMBEDDER_PATH)

    print(f"[DONE] Models saved to {MODELS_DIR}/")

if __name__ == "__main__":
    main()
