import os
import sys
import time
import cv2
import torch
from facenet_pytorch import MTCNN

# Add the root project dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.common import align_face, compute_blur_score, estimate_brightness, eye_angle_deg

# --- Config ---
BLUR_THRESHOLD = 35.0         # higher = sharper required
BRIGHTNESS_RANGE = (60, 200)  # acceptable brightness (V channel mean)
MAX_TILT_DEG = 20.0           # max allowed eye tilt
AUGMENT_HFLIP = True
MAX_IMAGES = 100
FACE_SIZE = (160, 160)
MIN_FACE_SIZE = 60

# --- Device ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True

print(f"[INFO] Using device: {device}")

# --- Init MTCNN ---
mtcnn = MTCNN(
    image_size=FACE_SIZE[0],
    margin=0,
    keep_all=True,
    device=device,
    post_process=False,
    min_face_size=MIN_FACE_SIZE
)

# --- Init user & dirs ---
name = input("Enter name: ").strip().replace(" ", "_")
img_dir = os.path.join("datasets", "faces", name)
os.makedirs(img_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

if not cap.isOpened():
    print("[ERROR] Webcam not found.")
    sys.exit(1)

print(f"[INFO] Get ready! Capturing faces for: {name}")
print("[INFO] Press 'q' to quit early.")

# --- Countdown with warm-up ---
for i in range(5, 0, -1):
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame during countdown.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Warm-up detect (forces CUDA kernels to load now)
    _ = mtcnn.detect(rgb)

    cv2.putText(frame, f"Starting in {i}...", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    cv2.imshow("Face Capture", frame)
    cv2.waitKey(1000)

count = 0

# --- Main loop ---
while count < MAX_IMAGES:
    ret, frame_bgr = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Detect with MTCNN
    boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
    detections = []

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(max(0, v)) for v in box]
            kp = {}
            if landmarks is not None:
                leye = landmarks[i][0]
                reye = landmarks[i][1]
                kp = {
                    "left_eye": (int(leye[0]), int(leye[1])),
                    "right_eye": (int(reye[0]), int(reye[1]))
                }
            detections.append({"box": (x1, y1, x2 - x1, y2 - y1), "keypoints": kp})

    # --- Process detections ---
    for det in detections:
        x, y, w, h = det['box']
        keypoints = det.get('keypoints', {})

        eyes_found = "left_eye" in keypoints and "right_eye" in keypoints
        aligned = None

        if eyes_found:
            try:
                aligned = align_face(frame_bgr, (x, y, w, h), keypoints, output_size=FACE_SIZE)
            except Exception:
                aligned = None

        if aligned is None:
            face_crop = frame_bgr[max(0, y):y + h, max(0, x):x + w]
            if face_crop.size == 0:
                continue
            aligned = cv2.resize(face_crop, FACE_SIZE)

        # --- Quality checks ---
        blur_val = compute_blur_score(aligned)
        bright_val = estimate_brightness(aligned)
        tilt_deg = abs(eye_angle_deg(keypoints)) if eyes_found else 0.0

        if blur_val < BLUR_THRESHOLD or not (BRIGHTNESS_RANGE[0] <= bright_val <= BRIGHTNESS_RANGE[1]):
            continue
        if eyes_found and tilt_deg > MAX_TILT_DEG:
            continue

        # --- Save aligned face ---
        img_filename = f"{name}_{count}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        cv2.imwrite(img_path, aligned)
        print(f"[INFO] Saved {img_filename} (blur={blur_val:.1f}, bright={bright_val:.1f}, tilt={tilt_deg:.1f})")
        count += 1

        # Optional horizontal flip
        if count < MAX_IMAGES and AUGMENT_HFLIP:
            flipped = cv2.flip(aligned, 1)
            img_filename = f"{name}_{count}.jpg"
            img_path = os.path.join(img_dir, img_filename)
            cv2.imwrite(img_path, flipped)
            print(f"[INFO] Saved augmented {img_filename}")
            count += 1

        if count >= MAX_IMAGES:
            break

    # Draw detections
    for det in detections:
        x, y, w, h = det['box']
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame_bgr, f"Captured: {count}/{MAX_IMAGES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Face Capture", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Early exit requested.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Collected {count} face images in: {img_dir}")
