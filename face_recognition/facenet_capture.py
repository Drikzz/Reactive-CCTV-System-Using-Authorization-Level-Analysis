import cv2
import os
import time
import sys
# Add the root project dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import align_face, compute_blur_score, estimate_brightness, eye_angle_deg

# --- Config (quality + augmentation) ---
BLUR_THRESHOLD = 35.0         # higher = sharper required
BRIGHTNESS_RANGE = (60, 200)  # acceptable brightness (V channel mean)
MAX_TILT_DEG = 20.0           # max allowed eye tilt
AUGMENT_HFLIP = True

# Initialize lightweight detectors (no embeddings)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# --- Init ---
name = input("Enter name: ").strip().replace(" ", "_")
img_dir = os.path.join("datasets", "faces", name)
os.makedirs(img_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

if not cap.isOpened():
    print("[ERROR] Webcam not found.")
    exit()

print(f"[INFO] Get ready! Capturing faces for: {name}")
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

    # Detect faces (and eyes) using Haar cascades
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

    detections = []
    for (x, y, w, h) in faces_rects:
        # ensure Python ints
        x, y, w, h = map(int, (x, y, w, h))
        keypoints = {}
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
        if len(eyes) >= 2:
            eyes2 = sorted(eyes, key=lambda e: int(e[2]) * int(e[3]), reverse=True)[:2]
            eyes2 = sorted(eyes2, key=lambda e: int(e[0]))
            (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes2
            left_center  = (int(x + ex1 + ew1 // 2), int(y + ey1 + eh1 // 2))
            right_center = (int(x + ex2 + ew2 // 2), int(y + ey2 + eh2 // 2))
            keypoints = {"left_eye": left_center, "right_eye": right_center}
        detections.append({"box": (x, y, w, h), "keypoints": keypoints})

    for det in detections:
        x, y, w, h = det['box']
        keypoints = det.get('keypoints', {})

        # Decide alignment strategy
        eyes_found = isinstance(keypoints, dict) and "left_eye" in keypoints and "right_eye" in keypoints
        if eyes_found:
            try:
                aligned = align_face(frame, (x, y, w, h), keypoints, output_size=FACE_SIZE)
            except Exception as e:
                # Alignment failed unexpectedly; fall back to simple crop
                aligned = None
        else:
            aligned = None

        if aligned is None:
            # Fallback: simple crop + resize (no rotation)
            face_crop = frame[max(0, y):max(0, y) + max(1, h), max(0, x):max(0, x) + max(1, w)]
            if face_crop.size == 0:
                continue
            aligned = cv2.resize(face_crop, FACE_SIZE)

        # Quality checks
        blur_val = compute_blur_score(aligned)
        bright_val = estimate_brightness(aligned)
        tilt_deg = abs(eye_angle_deg(keypoints)) if eyes_found else 0.0

        if blur_val < BLUR_THRESHOLD or not (BRIGHTNESS_RANGE[0] <= bright_val <= BRIGHTNESS_RANGE[1]):
            continue
        if eyes_found and tilt_deg > MAX_TILT_DEG:
            continue

        # Save aligned face image only (no embeddings)
        img_filename = f"{name}_{count}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        cv2.imwrite(img_path, aligned)
        print(f"[INFO] Saved {img_filename} (blur={blur_val:.1f}, bright={bright_val:.1f}" + (f", tilt={tilt_deg:.1f})" if eyes_found else ", tilt=NA)"))
        count += 1
      
        # Optional horizontal flip augmentation
        if count < MAX_IMAGES and AUGMENT_HFLIP:
            flipped = cv2.flip(aligned, 1)
            img_filename = f"{name}_{count}.jpg"
            img_path = os.path.join(img_dir, img_filename)
            cv2.imwrite(img_path, flipped)
            print(f"[INFO] Saved augmented {img_filename}")
            count += 1

        if count >= MAX_IMAGES:
            break

    # Draw detection boxes
    for det in detections:
        x, y, w, h = det['box']
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(0, int(w))
        h = max(0, int(h))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f"Captured: {count}/{MAX_IMAGES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Early exit requested.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Collected {count} face images in: {img_dir}")
