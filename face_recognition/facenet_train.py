import sys
import os
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# Add the root project dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from utils.common import load_known_faces
from utils.face_classifier import train_classifier, save_classifier_and_encoder
from utils.facenet_utils import get_embedder

DATASET_DIR = "datasets/faces"
FACE_SIZE = (160, 160)

def load_images_and_embeddings(dataset_dir, embedder, image_size=(160, 160)):
    X, y = [], []
    if not os.path.isdir(dataset_dir):
        return X, y

    # Each subfolder is a person/class
    for person in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(person_dir, fname)
            img = cv2.imread(fpath)
            if img is None or img.size == 0:
                continue

            # Images are already aligned crops; ensure size and compute embedding
            if (img.shape[1], img.shape[0]) != image_size:
                img = cv2.resize(img, image_size)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                emb = embedder.embeddings([rgb])[0]
            except Exception as e:
                print(f"[WARN] Embedding failed for {fpath}: {e}")
                continue
            X.append(emb)
            y.append(person)
    return X, y

embedder = get_embedder()
faces, names = load_images_and_embeddings(DATASET_DIR, embedder, FACE_SIZE)

if not faces or not names:
    print("[ERROR] No face images found. Capture faces first.")
    sys.exit(1)

counts = Counter(names)
if any(c < 2 for c in counts.values()) or len(counts) < 2:
    print("[WARN] Not enough samples per class for stratified split. Training on all data; skipping evaluation.")
    X_all = np.asarray(faces, dtype=np.float32)
    model, le = train_classifier(X_all, list(names))
    save_classifier_and_encoder(model, le, save_path='models/Facenet/')
    sys.exit(0)

X = np.asarray(faces, dtype=np.float32)
y_names = np.array(names)

# Try stratified splits; fallback to train-all if it fails
try:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_names, test_size=0.2, random_state=42, stratify=y_names
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )
except ValueError as e:
    print(f"[WARN] Stratified split failed: {e}. Training on all data; skipping evaluation.")
    model, le = train_classifier(X, list(y_names))
    save_classifier_and_encoder(model, le, save_path='models/Facenet/')
    sys.exit(0)

# Train on train set
model, le = train_classifier(X_train, list(y_train))
save_classifier_and_encoder(model, le, save_path='models/Facenet/')

# Evaluate on val and test
def eval_split(split_name, Xs, ys):
    y_true_enc = le.transform(ys)  # assumes all classes seen in training
    y_pred_enc = model.predict(Xs)
    report = classification_report(y_true_enc, y_pred_enc, target_names=list(le.classes_), digits=4, zero_division=0)
    cm = confusion_matrix(y_true_enc, y_pred_enc)
    print(f"[EVAL] {split_name} classification report:\n{report}")
    print(f"[EVAL] {split_name} confusion matrix:\n{cm}")
    return report, cm

val_report, val_cm = eval_split("Validation", X_val, y_val)
test_report, test_cm = eval_split("Test", X_test, y_test)

# Save metrics
os.makedirs('models', exist_ok=True)
with open(os.path.join('models', 'metrics.txt'), 'w') as f:
    f.write("=== Validation ===\n")
    f.write(val_report + "\n")
    f.write(str(val_cm) + "\n\n")
    f.write("=== Test ===\n")
    f.write(test_report + "\n")
    f.write(str(test_cm) + "\n")
print("[INFO] Metrics saved to models/metrics.txt")