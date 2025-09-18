import sys
import os
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add the root project dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.common import load_known_faces
from utils.face_classifier import train_classifier, save_classifier_and_encoder

faces, names = load_known_faces("datasets/faces")
if not faces:
    print("[ERROR] No face embeddings found. Capture faces first.")
    sys.exit(1)

# Ensure stratified split is feasible
counts = Counter(names)
if any(c < 2 for c in counts.values()) or len(counts) < 2:
    print("[WARN] Not enough samples per class for stratified split. Training on all data; skipping evaluation.")
    model, le = train_classifier(faces, names)
    save_classifier_and_encoder(model, le, save_path='models/Facenet/')
    sys.exit(0)

X = np.asarray(faces, dtype=np.float32)
y_names = np.array(names)

# Split: 20% test, then 20% of remaining for val
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y_names, test_size=0.2, random_state=42, stratify=y_names
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
)

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