import cv2
import os
import numpy as np
from datetime import datetime
from keras_facenet import FaceNet
import time

# --- Init ---
embedder = FaceNet()
name = input("Enter name: ").strip().replace(" ", "_")
img_dir = os.path.join("datasets", "faces", name)
os.makedirs(img_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

if not cap.isOpened():
    print("[ERROR] Webcam not found.")
    exit()

print(f"[INFO] Get ready! Capturing faces and embeddings for: {name}")
print("[INFO] Press 'q' to quit early.")
time.sleep(2)

count = 0
MAX_IMAGES = 100
FACE_SIZE = (160, 160)

# Countdown before starting
for i in range(3, 0, -1):
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame during countdown.")
        break
    cv2.putText(frame, f"Starting in {i}...", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    cv2.imshow("Face Capture", frame)
    cv2.waitKey(1000)

while count < MAX_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = embedder.extract(rgb, threshold=0.95)

    for det in detections:
        x, y, w, h = det['box']
        embedding = det['embedding']

        # Clamp coordinates within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(frame.shape[1] - x, w)
        h = min(frame.shape[0] - y, h)

        # Skip if face size is too small
        if w < 20 or h < 20:
            continue

        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        # Resize face and normalize it
        resized_face = cv2.resize(face_crop, FACE_SIZE)
        normalized_face = resized_face.astype('float32') / 255.0

        # Save face image
        img_filename = f"{name}_{count:03d}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        cv2.imwrite(img_path, (normalized_face * 255).astype('uint8'))

        # Save embedding
        emb_filename = f"{name}_{count:03d}.npy"
        emb_path = os.path.join(img_dir, emb_filename)
        np.save(emb_path, embedding)

        print(f"[INFO] Saved {img_filename} and its embedding")
        count += 1

        if count >= MAX_IMAGES:
            break

    # Draw detection boxes
    for det in detections:
        x, y, w, h = det['box']
        x = max(0, x)
        y = max(0, y)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f"Captured: {count}/{MAX_IMAGES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Early exit requested.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Collected {count} face images and embeddings in: {img_dir}")
