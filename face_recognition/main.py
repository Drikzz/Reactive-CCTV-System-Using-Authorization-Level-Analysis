import sys
import os

# Add root directory to path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from datetime import datetime
import numpy as np
from ultralytics import YOLO

# FaceNet utils
try:
	from utils.face_classifier import load_classifier_and_encoder
	from utils.facenet_utils import recognize_faces
	FACE_RECOGNITION_ENABLED = True
except Exception as e:
	print(f"[ERROR] Couldn't import face recognition utils: {e}")
	FACE_RECOGNITION_ENABLED = False

# --- Config --- 
USE_WEBCAM = True
VIDEO_PATH = "sample_video.mp4"
MODEL_PATH = "models/YOLOv8/yolov8n.pt"
# Updated output directory structure
OUTPUT_DIR = "outputs"
ANNOTATED_FRAMES_DIR = os.path.join(OUTPUT_DIR, "annotated_frames")
CROPPED_FACES_DIR = os.path.join(OUTPUT_DIR, "cropped_faces")
SAVE_FACES = True
RESIZE_WIDTH = 640
RECOG_THRESHOLD = 0.65  # probability threshold to accept identity
# -------------- 

model = YOLO(MODEL_PATH)

classifier, label_encoder, centroids = None, None, None
# dist_threshold = 1.0  # Default if not loaded from file
dist_threshold = 0.8  # hardcoded

if FACE_RECOGNITION_ENABLED:
    try:
        classifier, label_encoder, centroids = load_classifier_and_encoder(model_path='models/Facenet/')
        
        # Load distance threshold if available
        threshold_path = os.path.join('models/Facenet/', 'distance_threshold.npy')
        if os.path.exists(threshold_path):
            dist_threshold = float(np.load(threshold_path))
            print(f"[INFO] Loaded distance threshold: {dist_threshold:.3f}")
        else:
            print(f"[INFO] Using default distance threshold: {dist_threshold:.3f}")
            
        print(f"[INFO] Loaded classifier with {len(label_encoder.classes_)} known identities")
        if centroids:
            print(f"[INFO] Loaded {len(centroids)} class centroids")
    except Exception as e:
        print(f"[ERROR] Failed to load classifier or encoder: {e}")
        FACE_RECOGNITION_ENABLED = False

cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)

# Create output folders if saving faces
if SAVE_FACES:
    # Create all necessary directories
    os.makedirs(os.path.join(ANNOTATED_FRAMES_DIR, "known"), exist_ok=True)
    os.makedirs(os.path.join(ANNOTATED_FRAMES_DIR, "unknown"), exist_ok=True)
    os.makedirs(os.path.join(CROPPED_FACES_DIR, "known"), exist_ok=True)
    os.makedirs(os.path.join(CROPPED_FACES_DIR, "unknown"), exist_ok=True)

frame_num = 0
print("[INFO] Starting detection" + (" + face recognition" if FACE_RECOGNITION_ENABLED else ""))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of stream or camera not found.")
        break

    # Keep original for face recognition
    original_frame = frame.copy()
    orig_h, orig_w = original_frame.shape[:2]

    # Resize for YOLO (speed)
    ratio = RESIZE_WIDTH / orig_w
    resized_frame = cv2.resize(original_frame, (RESIZE_WIDTH, int(orig_h * ratio)))
    annotated_frame = resized_frame.copy()

    # --- YOLO detection on resized ---
    results = model(resized_frame)
    people_boxes = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            people_boxes.append((x1, y1, x2, y2))

    # --- Face Recognition or fallback ---
    if FACE_RECOGNITION_ENABLED:
        try:
            faces = recognize_faces(original_frame, classifier, label_encoder, 
                                   threshold=RECOG_THRESHOLD, 
                                   centroids=centroids,
                                   dist_threshold=dist_threshold)
            for face in faces:
                name = face['name']
                prob = face.get('prob', 0.0)
                distance = face.get('distance', None)
                x1o, y1o, x2o, y2o = face['bbox']

                # Scale face coords to resized frame
                x1_r = int(x1o * ratio)
                y1_r = int(y1o * ratio)
                x2_r = int(x2o * ratio)
                y2_r = int(y2o * ratio)
                face_center = ((x1_r + x2_r) // 2, (y1_r + y2_r) // 2)

                # Save cropped face (both known and unknown)
                if SAVE_FACES:
                    face_crop = original_frame[y1o:y2o, x1o:x2o]
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
                    
                    # Save to appropriate directory based on recognition result
                    if name != "Unknown":
                        filename = f"{name}_{timestamp}.jpg"
                        save_path = os.path.join(CROPPED_FACES_DIR, "known", filename)
                    else:
                        filename = f"unknown_{timestamp}.jpg"
                        save_path = os.path.join(CROPPED_FACES_DIR, "unknown", filename)
                    
                    cv2.imwrite(save_path, face_crop)

                matched = False
                for (x1, y1, x2, y2) in people_boxes:
                    if x1 <= face_center[0] <= x2 and y1 <= face_center[1] <= y2:
                        # Simplified display - only show name, no probabilities
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, name, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        matched = True
                        break

                if not matched:
                    # Simplified display - only show name, no probabilities
                    color = (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1_r, y1_r), (x2_r, y2_r), color, 2)
                    cv2.putText(annotated_frame, name, (x1_r, y1_r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except Exception as e:
            print(f"[ERROR] Face recognition failed: {e}")
            # Fall back to YOLO detections
            for (x1, y1, x2, y2) in people_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red border
                cv2.putText(annotated_frame, "Unknown", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Save cropped image of the unknown face
                if SAVE_FACES:
                    x1o = max(0, int(x1 / ratio))
                    y1o = max(0, int(y1 / ratio))
                    x2o = min(orig_w - 1, int(x2 / ratio))
                    y2o = min(orig_h - 1, int(y2 / ratio))
                    if x2o > x1o and y2o > y1o:
                        unknown_face = original_frame[y1o:y2o, x1o:x2o]
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
                        filename = f"unknown_{timestamp}.jpg"
                        save_path = os.path.join(CROPPED_FACES_DIR, "unknown", filename)
                        cv2.imwrite(save_path, unknown_face)

    # --- Save Full Annotated Frame (with bounding boxes) ---
    if SAVE_FACES:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        frame_filename = f"frame_{frame_num}_{timestamp}.jpg"
        
        # Determine if this frame contains known or unknown faces
        has_known_face = False
        if FACE_RECOGNITION_ENABLED:
            try:
                has_known_face = any(face['name'] != "Unknown" for face in faces)
            except:
                pass
        
        # Save to appropriate directory
        if has_known_face:
            frame_path = os.path.join(ANNOTATED_FRAMES_DIR, "known", frame_filename)
        else:
            frame_path = os.path.join(ANNOTATED_FRAMES_DIR, "unknown", frame_filename)
        
        cv2.imwrite(frame_path, annotated_frame)

    # --- Display ---
    if USE_WEBCAM:
        cv2.imshow("YOLOv8 + Face Recognition", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    frame_num += 1

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Processed {frame_num} frames.")
