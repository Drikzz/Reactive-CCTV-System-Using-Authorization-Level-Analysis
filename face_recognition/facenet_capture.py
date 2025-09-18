import cv2
import os
import numpy as np
from datetime import datetime
import time
import sys
# Add the root project dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.facenet_utils import get_embedder
from utils.common import align_face, compute_blur_score, estimate_brightness, eye_angle_deg

# --- Config (quality + augmentation) ---
BLUR_THRESHOLD = 35.0         # higher = sharper required
BRIGHTNESS_RANGE = (60, 200)  # acceptable brightness (V channel mean)
MAX_TILT_DEG = 20.0           # max allowed eye tilt
AUGMENT_HFLIP = True

# --- Init ---
embedder = get_embedder()
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
        keypoints = det.get('keypoints', {})
        # Align face crop using eyes, fallback to simple resize
        aligned = align_face(frame, (x, y, w, h), keypoints, output_size=FACE_SIZE)
        if aligned is None:
            face_crop = frame[max(0,y):max(0,y)+max(1,h), max(0,x):max(0,x)+max(1,w)]
            if face_crop.size == 0:
                continue
            aligned = cv2.resize(face_crop, FACE_SIZE)

        # Quality checks
        blur_val = compute_blur_score(aligned)
        bright_val = estimate_brightness(aligned)
        tilt_deg = abs(eye_angle_deg(keypoints))

        if blur_val < BLUR_THRESHOLD or not (BRIGHTNESS_RANGE[0] <= bright_val <= BRIGHTNESS_RANGE[1]) or tilt_deg > MAX_TILT_DEG:
            # Skip low-quality sample
            continue

        # Compute embedding from aligned face
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        emb = embedder.embeddings([aligned_rgb])[0]

        # Save face image
        img_filename = f"{name}_{count:03d}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        cv2.imwrite(img_path, aligned)

        # Save embedding
        emb_filename = f"{name}_{count:03d}.npy"
        emb_path = os.path.join(img_dir, emb_filename)
        np.save(emb_path, emb)

        print(f"[INFO] Saved {img_filename} (blur={blur_val:.1f}, bright={bright_val:.1f}, tilt={tilt_deg:.1f})")
        count += 1

        # Optional horizontal flip augmentation
        if count < MAX_IMAGES and AUGMENT_HFLIP:
            flipped = cv2.flip(aligned, 1)
            flipped_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
            emb_flip = embedder.embeddings([flipped_rgb])[0]

            img_filename = f"{name}_{count:03d}.jpg"
            img_path = os.path.join(img_dir, img_filename)
            cv2.imwrite(img_path, flipped)

            emb_filename = f"{name}_{count:03d}.npy"
            emb_path = os.path.join(img_dir, emb_filename)
            np.save(emb_path, emb_flip)

            print(f"[INFO] Saved augmented {img_filename}")
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
