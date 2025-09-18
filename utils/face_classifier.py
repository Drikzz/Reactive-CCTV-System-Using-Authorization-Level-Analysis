import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

def build_svm_pipeline():
    """Linear SVM with calibrated probabilities via probability=True."""
    return make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    )

def train_classifier(encodings, names):
    """Train SVM classifier and return the model and label encoder."""
    # Normalize inputs
    X = np.asarray(encodings, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    y_names = list(names)

    # Validation without ambiguous truth checks
    if X.shape[0] == 0 or len(y_names) == 0 or X.shape[0] != len(y_names):
        raise ValueError("No data or mismatched encodings/names to train SVM.")
    if len(set(y_names)) < 2:
        raise ValueError("At least two different people are required to train an SVM classifier.")

    le = LabelEncoder()
    y = le.fit_transform(y_names)

    model = build_svm_pipeline()
    model.fit(X, y)
    return model, le

def save_classifier_and_encoder(model, le, save_path='models/Facenet/'):
    """Save the trained model and label encoder to the specified path."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    joblib.dump(model, os.path.join(save_path, 'svm_classifier.pkl'))
    joblib.dump(le, os.path.join(save_path, 'label_encoder.pkl'))
    print(f"[INFO] Model and encoder saved to: {save_path}")

def train_and_save_svm_classifier(encodings, names, save_path='models/Facenet/'):
    """Train the SVM classifier and save the model and encoder."""
    model, le = train_classifier(encodings, names)
    save_classifier_and_encoder(model, le, save_path)

def load_classifier_and_encoder(model_path='models/Facenet/'):
    """Load the classifier and label encoder from the specified path."""
    classifier = joblib.load(os.path.join(model_path, 'svm_classifier.pkl'))
    label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
    return classifier, label_encoder