import os
import cv2
from datetime import datetime
from ultralytics import YOLO

# FaceNet utils
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
ANNOTATED_FRAME_DIR = "outputs/annotated_frames/known"  # Full frame saves here
LOGS_DIR = "logs/unknown"
SAVE_FACES = True
RESIZE_WIDTH = 640
# -------------- 

model = YOLO(MODEL_PATH)

classifier, label_encoder = None, None
if FACE_RECOGNITION_ENABLED:
    try:
        classifier, label_encoder = load_classifier_and_encoder(model_path='models')
    except Exception as e:
        print(f"[ERROR] Failed to load classifier or encoder: {e}")
        FACE_RECOGNITION_ENABLED = False

cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)

# Create output folders if saving faces
if SAVE_FACES:
    os.makedirs(ANNOTATED_FRAME_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

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
            faces = recognize_faces(original_frame, classifier, label_encoder)
            for face in faces:
                name = face['name']
                top, right, bottom, left = face['location']

                # Scale face coords to resized frame
                top_r = int(top * ratio)
                right_r = int(right * ratio)
                bottom_r = int(bottom * ratio)
                left_r = int(left * ratio)
                face_center = ((left_r + right_r) // 2, (top_r + bottom_r) // 2)

                matched = False
                for (x1, y1, x2, y2) in people_boxes:
                    if x1 <= face_center[0] <= x2 and y1 <= face_center[1] <= y2:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, name, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        matched = True
                        break

                if not matched:
                    cv2.rectangle(annotated_frame, (left_r, top_r), (right_r, bottom_r), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, name, (left_r, top_r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception as e:
            print(f"[ERROR] Face recognition failed: {e}")
            FACE_RECOGNITION_ENABLED = False

    else:
        # --- Fallback for unknown faces if recognition is disabled ---
        # Use YOLO detections for the bounding boxes (red border for unknown people)
        for (x1, y1, x2, y2) in people_boxes:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red border
            cv2.putText(annotated_frame, "Unknown", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Save cropped image of the unknown face
            if SAVE_FACES:
                unknown_face = original_frame[y1:y2, x1:x2]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
                filename = f"unknown_{timestamp}.jpg"
                save_path = os.path.join(LOGS_DIR, filename)
                cv2.imwrite(save_path, unknown_face)

    # --- Save Full Annotated Frame (with bounding boxes) ---
    if SAVE_FACES:
        # Save the full annotated frame (with bounding boxes) in 'annotated_frames/known'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        frame_filename = f"frame_{frame_num}_{timestamp}.jpg"
        frame_path = os.path.join(ANNOTATED_FRAME_DIR, frame_filename)
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
