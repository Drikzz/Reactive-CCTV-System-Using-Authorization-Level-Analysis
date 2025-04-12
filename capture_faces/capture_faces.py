import cv2
import os
import numpy as np
from datetime import datetime
from keras_facenet import FaceNet

# Setup FaceNet embedder
embedder = FaceNet()

# Ask user for name
name = input("Enter name: ").strip().replace(" ", "_")
img_dir = os.path.join("datasets", "faces", name)
os.makedirs(img_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # Ensure RGB

if not cap.isOpened():
    print("[ERROR] Webcam not found.")
    exit()

print(f"[INFO] Capturing faces and embeddings for: {name}")
print("[INFO] Press 'q' to quit early.")

count = 0
MAX_IMAGES = 100

while count < MAX_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = embedder.extract(rgb, threshold=0.95)

    for det in detections:
        box = det['box']
        embedding = det['embedding']
        x, y, w, h = box
        face_crop = frame[y:y+h, x:x+w]

        if face_crop.size == 0:
            continue

        # Save image
        img_filename = f"{name}_{count:03d}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        cv2.imwrite(img_path, face_crop)

        # Save embedding
        emb_filename = f"{name}_{count:03d}.npy"
        emb_path = os.path.join(img_dir, emb_filename)
        np.save(emb_path, embedding)

        print(f"[INFO] Saved {img_filename} and its embedding")
        count += 1

        if count >= MAX_IMAGES:
            break

    # Draw boxes
    for det in detections:
        x, y, w, h = det['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Collected {count} face images and embeddings in: {img_dir}")
