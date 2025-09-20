import sys
import os

# Add root directory to path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from datetime import datetime, timedelta
import numpy as np
from ultralytics import YOLO
import csv
from threading import Thread, Event
from queue import Queue, Empty

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
# Updated output directory structure (new)
LOGS_BASE = os.path.join("logs", "FaceNet")
ANNOTATED_BASE = os.path.join("annotated_frames", "FaceNet")  # changed from "annotated" to "annotated_frames"
LOGS_KNOWN_DIR = os.path.join(LOGS_BASE, "known")
LOGS_UNKNOWN_DIR = os.path.join(LOGS_BASE, "unknown")
ANNOTATED_KNOWN_DIR = os.path.join(ANNOTATED_BASE, "known")
ANNOTATED_UNKNOWN_DIR = os.path.join(ANNOTATED_BASE, "unknown")

SAVE_FACES = True
RESIZE_WIDTH = 640
RECOG_THRESHOLD = 0.65  # probability threshold to accept identity
# Threading/throughput
CAPTURE_QUEUE_SIZE = 4
DISPLAY_QUEUE_SIZE = 2
KNOWN_SAVE_INTERVAL_MIN = 5  # save known-frame only first time or every N minutes
# -------------- 

model = YOLO(MODEL_PATH)

classifier, label_encoder, centroids = None, None, None
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

# --- New: ensure directories for logs and annotated artifacts ---
if SAVE_FACES:
    for d in [LOGS_KNOWN_DIR, LOGS_UNKNOWN_DIR, ANNOTATED_KNOWN_DIR, ANNOTATED_UNKNOWN_DIR]:
        os.makedirs(d, exist_ok=True)
# Ensure unknown annotated path (new structure)
os.makedirs(ANNOTATED_UNKNOWN_DIR, exist_ok=True)

# --- New: CSV logging helper ---
def append_csv(csv_path, header, row):
    new_file = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file and header:
            w.writerow(header)
        w.writerow(row)

# --- New: frame grabber and processor threads ---
def grab_frames(cap, frame_q, stop_event):
    frame_num = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        ts = datetime.now()
        try:
            frame_q.put((frame_num, ts, frame), timeout=0.05)
            frame_num += 1
        except:
            # queue full; drop frame to keep UI smooth
            pass
    cap.release()

def process_frames(frame_q, display_q, stop_event):
    processed = 0
    known_last_saved = {}  # name -> datetime of last saved annotated frame

    while not stop_event.is_set() or not frame_q.empty():
        try:
            frame_num, ts, frame = frame_q.get(timeout=0.05)
        except Empty:
            continue

        original_frame = frame.copy()
        orig_h, orig_w = original_frame.shape[:2]

        # Resize for YOLO (speed)
        ratio = RESIZE_WIDTH / orig_w
        resized_frame = cv2.resize(original_frame, (RESIZE_WIDTH, int(orig_h * ratio)))

        # Annotate always on full-res for better saved context
        annotated_frame = original_frame.copy()

        # --- YOLO detection on resized ---
        try:
            results = model(resized_frame)
        except Exception as e:
            print(f"[ERROR] YOLO inference failed: {e}")
            results = []

        people_boxes_resized = []
        if results:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    people_boxes_resized.append((x1, y1, x2, y2))

        # Map person boxes back to original coords
        people_boxes = []
        for (x1, y1, x2, y2) in people_boxes_resized:
            x1o = max(0, int(x1 / ratio))
            y1o = max(0, int(y1 / ratio))
            x2o = min(orig_w - 1, int(x2 / ratio))
            y2o = min(orig_h - 1, int(y2 / ratio))
            if x2o > x1o and y2o > y1o:
                people_boxes.append((x1o, y1o, x2o, y2o))

        faces = []
        unknown_in_frame = False
        known_names_in_frame = set()
        faces_by_name = {}  # name -> list of face dicts

        if FACE_RECOGNITION_ENABLED and people_boxes:
            # --- Run FaceNet only inside person ROIs ---
            for (px1, py1, px2, py2) in people_boxes:
                roi = original_frame[py1:py2, px1:px2]
                if roi.size == 0:
                    continue
                try:
                    faces_roi = recognize_faces(
                        roi, classifier, label_encoder,
                        threshold=RECOG_THRESHOLD,
                        centroids=centroids,
                        dist_threshold=dist_threshold
                    )
                except Exception as e:
                    print(f"[ERROR] Face recognition failed on ROI: {e}")
                    faces_roi = []

                # Offset bbox from ROI -> original
                for f in faces_roi:
                    bx1, by1, bx2, by2 = f['bbox']
                    f_global = dict(f)
                    f_global['bbox'] = (px1 + bx1, py1 + by1, px1 + bx2, py1 + by2) # noqa: F841
                    faces.append(f_global)

        elif FACE_RECOGNITION_ENABLED and not people_boxes:
            # No persons detected; optionally skip recognition to save compute
            faces = []
        else:
            # No FaceNet: fallback — draw person boxes as unknowns
            for (px1, py1, px2, py2) in people_boxes:
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "Unknown", (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Always save cropped region and log as unknown (fallback)
                ts_str = ts.strftime('%Y%m%d_%H%M%S%f')
                face_path = os.path.join(LOGS_UNKNOWN_DIR, f"unknown_{ts_str}.jpg")
                try:
                    cv2.imwrite(face_path, original_frame[py1:py2, px1:px2])
                    append_csv(
                        os.path.join(LOGS_UNKNOWN_DIR, "detections.csv"),
                        header=["timestamp", "frame", "file", "x1", "y1", "x2", "y2", "prob", "distance"],
                        row=[ts.isoformat(), frame_num, os.path.basename(face_path), px1, py1, px2, py2, "", ""]
                    )
                except Exception as e:
                    print(f"[WARN] Failed saving unknown fallback crop: {e}")
                unknown_in_frame = True

        # Draw and persist outputs when FaceNet is available
        for f in faces:
            name = f.get('name', 'Unknown')
            prob = float(f.get('prob', 0.0) or 0.0)
            distance = f.get('distance', None)
            x1o, y1o, x2o, y2o = f['bbox']
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Draw face box + label (no person box to avoid duplicates)
            cv2.rectangle(annotated_frame, (x1o, y1o), (x2o, y2o), color, 2)
            cv2.putText(annotated_frame, name, (x1o, max(0, y1o - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if name == "Unknown":
                unknown_in_frame = True
                # Always save cropped unknown face and log
                ts_str = ts.strftime('%Y%m%d_%H%M%S%f')
                face_path = os.path.join(LOGS_UNKNOWN_DIR, f"unknown_{ts_str}.jpg")
                try:
                    face_crop = original_frame[max(0, y1o):min(orig_h, y2o), max(0, x1o):min(orig_w, x2o)]
                    if face_crop.size:
                        cv2.imwrite(face_path, face_crop)
                    append_csv(
                        os.path.join(LOGS_UNKNOWN_DIR, "detections.csv"),
                        header=["timestamp", "frame", "file", "x1", "y1", "x2", "y2", "prob", "distance"],
                        row=[ts.isoformat(), frame_num, os.path.basename(face_path), x1o, y1o, x2o, y2o, f"{prob:.3f}", "" if distance is None else f"{float(distance):.4f}"]
                    )
                except Exception as e:
                    print(f"[WARN] Failed saving/logging unknown face: {e}")
            else:
                known_names_in_frame.add(name)
                faces_by_name.setdefault(name, []).append(f)

        # Always save one annotated full frame when unknowns are present, to outputs/annotated_frames/unknown
        if unknown_in_frame:
            ts_str = ts.strftime('%Y%m%d_%H%M%S%f')
            frame_path = os.path.join(ANNOTATED_UNKNOWN_DIR, f"frame_{frame_num}_{ts_str}.jpg")  # replaced ANNOTATED_FRAMES_DIR/.../unknown -> ANNOTATED_UNKNOWN_DIR
            try:
                cv2.imwrite(frame_path, annotated_frame)
            except Exception as e:
                print(f"[WARN] Failed saving unknown annotated frame: {e}")

        # Known: append to CSV and save annotated frame only when first seen or every KNOWN_SAVE_INTERVAL_MIN
        if known_names_in_frame:
            now = ts
            due_to_save = False
            for name in list(known_names_in_frame):
                last = known_last_saved.get(name)
                if last is None or now - last >= timedelta(minutes=KNOWN_SAVE_INTERVAL_MIN):
                    known_last_saved[name] = now
                    # Log this known person now (may have multiple faces in this frame)
                    for f in faces_by_name.get(name, []):
                        x1o, y1o, x2o, y2o = f['bbox']
                        prob = float(f.get('prob', 0.0) or 0.0)
                        distance = f.get('distance', None)
                        append_csv(
                            os.path.join(LOGS_KNOWN_DIR, "detections.csv"),
                            header=["timestamp", "frame", "name", "x1", "y1", "x2", "y2", "prob", "distance"],
                            row=[ts.isoformat(), frame_num, name, x1o, y1o, x2o, y2o, f"{prob:.3f}", "" if distance is None else f"{float(distance):.4f}"]
                        )
                    due_to_save = True
            if SAVE_FACES and due_to_save:
                ts_str = ts.strftime('%Y%m%d_%H%M%S%f')
                frame_path = os.path.join(ANNOTATED_KNOWN_DIR, f"frame_{frame_num}_{ts_str}.jpg")
                try:
                    cv2.imwrite(frame_path, annotated_frame)
                except Exception as e:
                    print(f"[WARN] Failed saving known annotated frame: {e}")

        processed += 1

        # Push latest annotated frame for display (drop-old strategy)
        try:
            display_q.put((frame_num, ts, annotated_frame), timeout=0.01)
        except:
            try:
                _ = display_q.get_nowait()
                display_q.put((frame_num, ts, annotated_frame), timeout=0.01)
            except:
                pass

    print(f"[INFO] Processed {processed} frames.")

# --- Main: start threads and display loop ---
cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)

frame_q = Queue(maxsize=CAPTURE_QUEUE_SIZE)
display_q = Queue(maxsize=DISPLAY_QUEUE_SIZE)
stop_event = Event()

print("[INFO] Starting detection" + (" + face recognition (ROI)" if FACE_RECOGNITION_ENABLED else ""))

grab_t = Thread(target=grab_frames, args=(cap, frame_q, stop_event), daemon=True)
proc_t = Thread(target=process_frames, args=(frame_q, display_q, stop_event), daemon=True)
grab_t.start()
proc_t.start()

# Display in main thread
try:
    while not stop_event.is_set():
        try:
            frame_num, ts, annotated = display_q.get(timeout=0.05)
        except Empty:
            if not grab_t.is_alive() and not proc_t.is_alive():
                break
            # keep looping
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
            continue

        if USE_WEBCAM:
            cv2.imshow("YOLOv8 + Face Recognition (Async ROI)", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        else:
            # If running on file, we can still show or just progress
            pass
finally:
    stop_event.set()
    grab_t.join(timeout=1.0)
    proc_t.join(timeout=2.0)
    cv2.destroyAllWindows()
print(f"[INFO] Processed {frame_num} frames.")