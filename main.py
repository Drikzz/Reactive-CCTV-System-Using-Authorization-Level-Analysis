import os
import cv2
from datetime import datetime
from ultralytics import YOLO

# Import face recognition utilities
try:
    from utils.face_classifier import load_classifier_and_encoder, recognize_faces
    FACE_RECOGNITION_ENABLED = True
except Exception as e:
    print(f"[ERROR] Couldn't import face_utils: {e}")
    FACE_RECOGNITION_ENABLED = False

# --- Config ---
USE_WEBCAM = True
VIDEO_PATH = "sample_video.mp4"
MODEL_PATH = "models/YOLOv8/yolov8n.pt"
SAVE_DIR = "outputs/annotated_frames"
# --------------

# Load YOLOv8
model = YOLO(MODEL_PATH)

# Load trained face recognition model
classifier, label_encoder = None, None
if FACE_RECOGNITION_ENABLED:
    try:
        classifier, label_encoder = load_classifier_and_encoder(model_path='models')
    except Exception as e:
        print(f"[ERROR] Failed to load classifier or encoder: {e}")
        FACE_RECOGNITION_ENABLED = False

# Set up video
cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)
os.makedirs(SAVE_DIR, exist_ok=True)

frame_num = 0
print("[INFO] Starting detection" + (" + face recognition" if FACE_RECOGNITION_ENABLED else ""))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of stream or camera not found.")
        break

    results = model(frame)
    print(results)
    annotated_frame = results[0].plot()

    if FACE_RECOGNITION_ENABLED:
        try:
            faces = recognize_faces(frame, classifier, label_encoder)
            print(faces)
            for face in faces:
                name = face['name']
                top, right, bottom, left = face['location']

                cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(annotated_frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception as e:
            print(f"[ERROR] Face recognition failed: {e}")
            FACE_RECOGNITION_ENABLED = False

    # Save frame
    filename = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{frame_num:04d}.jpg"
    cv2.imwrite(os.path.join(SAVE_DIR, filename), annotated_frame)

    if USE_WEBCAM:
        cv2.imshow("YOLOv8 + Face Recognition", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Saved {frame_num} frames to '{SAVE_DIR}'")
