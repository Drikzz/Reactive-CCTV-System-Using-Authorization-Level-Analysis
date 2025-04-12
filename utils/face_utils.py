import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
import joblib

embedder = FaceNet()

def load_known_faces(folder="datasets/faces"):
    encodings = []
    names = []

    for person_name in os.listdir(folder):
        person_dir = os.path.join(folder, person_name)
        if not os.path.isdir(person_dir):
            continue

        for file in os.listdir(person_dir):
            path = os.path.join(person_dir, file)
            image = cv2.imread(path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = embedder.extract(image_rgb, threshold=0.95)

            if detections:
                embedding = detections[0]['embedding']
                encodings.append(embedding)
                names.append(person_name)

    print(f"[INFO] Loaded {len(encodings)} face encodings.")
    return encodings, names

def train_and_save_svm_classifier(encodings, names, save_path='models'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    le = LabelEncoder()
    labels = le.fit_transform(names)

    model = make_pipeline(SVC(kernel='linear', probability=True))
    model.fit(encodings, labels)

    joblib.dump(model, os.path.join(save_path, 'svm_classifier.pkl'))
    joblib.dump(le, os.path.join(save_path, 'label_encoder.pkl'))

    print("[INFO] SVM Classifier trained and saved.")

def recognize_faces(frame, classifier, label_encoder, threshold=0.5):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = embedder.extract(rgb, threshold=0.95)
    recognized = []

    for det in detections:
        embedding = det['embedding']
        box = det['box']
        name = "Unknown"

        probs = classifier.predict_proba([embedding])
        max_prob = np.max(probs)
        pred = classifier.predict([embedding])[0]

        if max_prob > threshold:
            name = label_encoder.inverse_transform([pred])[0]

        recognized.append({"name": name, "location": box})

    return recognized
