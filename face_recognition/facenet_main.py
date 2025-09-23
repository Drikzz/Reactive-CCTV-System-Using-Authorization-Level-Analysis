import sys
import os
import csv
import cv2
import torch
import numpy as np
from datetime import datetime, timedelta
from threading import Thread, Event
from queue import Queue, Empty
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization

# Add root directory to sys.path for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Face classifier utils ---
try:
    from utils.face_classifier import load_classifier_and_encoder
    FACE_RECOGNITION_ENABLED = True
except Exception as e:
    print(f"[ERROR] Couldn't import classifier utils: {e}")
    FACE_RECOGNITION_ENABLED = False

# --- Config ---
USE_WEBCAM = True
VIDEO_PATH = "sample_video.mp4"

# Output directories
LOGS_BASE = os.path.join("logs", "FaceNet")
ANNOTATED_BASE = os.path.join("annotated_frames", "FaceNet")
LOGS_KNOWN_DIR = os.path.join(LOGS_BASE, "known")
LOGS_UNKNOWN_DIR = os.path.join(LOGS_BASE, "unknown")
ANNOTATED_KNOWN_DIR = os.path.join(ANNOTATED_BASE, "known")
ANNOTATED_UNKNOWN_DIR = os.path.join(ANNOTATED_BASE, "unknown")

SAVE_FACES = True
RESIZE_WIDTH = 640
RECOG_THRESHOLD = 0.65
CAPTURE_QUEUE_SIZE = 4
DISPLAY_QUEUE_SIZE = 2
KNOWN_SAVE_INTERVAL_MIN = 5

# Device + models
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=device, post_process=False)
embedder = InceptionResnetV1(pretrained="vggface2").to(device).eval()

classifier, label_encoder, centroids = None, None, None
dist_threshold = 0.8  # default

import joblib

# --- Load models produced by the new facenet_train.py ---
if FACE_RECOGNITION_ENABLED:
    try:
        models_dir = os.path.join("models", "FaceNet")
        clf_path = os.path.join(models_dir, "facenet_svm.joblib")
        le_path = os.path.join(models_dir, "label_encoder.joblib")
        thr_path = os.path.join(models_dir, "distance_threshold.npy")
        embedder_state_path = os.path.join(models_dir, "inception_resnet_v1.pt")

        # load SVM and label encoder (joblib)
        if not os.path.exists(clf_path) or not os.path.exists(le_path):
            raise FileNotFoundError(f"Missing classifier or encoder in {models_dir}")

        classifier = joblib.load(clf_path)
        label_encoder = joblib.load(le_path)
        centroids = None  # centroids weren't saved by the new training script

        # load threshold (.npy)
        if os.path.exists(thr_path):
            dist_threshold = float(np.load(thr_path))
            print(f"[INFO] Loaded distance threshold (npy): {dist_threshold:.3f}")
        else:
            print(f"[INFO] distance_threshold.npy not found — using default {dist_threshold:.3f}")

        print(f"[INFO] Loaded classifier from: {clf_path}")
        print(f"[INFO] Label encoder classes: {list(label_encoder.classes_)}")
        # Inspect classifier classes_ (should be encoded ints)
        print(f"[INFO] Classifier classes_: {getattr(classifier, 'classes_', 'N/A')}")

    except Exception as e:
        print(f"[ERROR] Failed to load classifier/encoder/threshold: {e}")
        FACE_RECOGNITION_ENABLED = False

# Ensure dirs exist
if SAVE_FACES:
    for d in [LOGS_KNOWN_DIR, LOGS_UNKNOWN_DIR, ANNOTATED_KNOWN_DIR, ANNOTATED_UNKNOWN_DIR]:
        os.makedirs(d, exist_ok=True)

# --- Helpers ---
def append_csv(csv_path, header, row):
    new_file = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file and header:
            w.writerow(header)
        w.writerow(row)

def compute_embedding_distance(embedding, centroid):
    return float(np.linalg.norm(np.asarray(embedding, dtype=np.float32) - np.asarray(centroid, dtype=np.float32)))

# --- Frame grabber ---
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
            pass
    cap.release()

# --- Frame processor ---
def process_frames(frame_q, display_q, stop_event):
    processed = 0
    known_last_saved = {}

    while not stop_event.is_set() or not frame_q.empty():
        try:
            frame_num, ts, frame = frame_q.get(timeout=0.05)
        except Empty:
            continue

        original_frame = frame.copy()
        orig_h, orig_w = original_frame.shape[:2]
        annotated_frame = original_frame.copy()

        # Resize (optional)
        ratio = 1.0
        if RESIZE_WIDTH and orig_w > RESIZE_WIDTH:
            ratio = RESIZE_WIDTH / orig_w
            frame_proc = cv2.resize(original_frame, (RESIZE_WIDTH, int(orig_h * ratio)))
        else:
            frame_proc = original_frame

        # Detect faces
        rgb_proc = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(rgb_proc)

        print(f"[DEBUG] MTCNN boxes: {boxes} probs: {probs}")

        try:
            boxes, probs = mtcnn.detect(rgb_proc)
        except Exception as e:
            print(f"[ERROR] MTCNN failed: {e}")

        faces, unknown_in_frame, known_names_in_frame, faces_by_name = [], False, set(), {}

        if FACE_RECOGNITION_ENABLED and boxes is not None:
            # Scale boxes back
            boxes_orig = []
            for b in boxes:
                x1, y1, x2, y2 = b
                x1o, y1o, x2o, y2o = int(x1 / ratio), int(y1 / ratio), int(x2 / ratio), int(y2 / ratio)
                if x2o > x1o and y2o > y1o:
                    boxes_orig.append([x1o, y1o, x2o, y2o])

            if boxes_orig:
                rgb_full = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                try:
                    aligned_batch = mtcnn.extract(rgb_full, boxes_orig, save_path=None)
                except Exception as e:
                    print(f"[ERROR] MTCNN extract failed: {e}")
                    aligned_batch = None

                if aligned_batch is not None:
                    with torch.no_grad():
                        aligned_std = fixed_image_standardization(aligned_batch.to(device))
                        embeddings = embedder(aligned_std).detach().cpu().numpy()

                    # Classify embeddings
                    for i, emb in enumerate(embeddings):
                        x1o, y1o, x2o, y2o = map(int, boxes_orig[i])
                        try:
                            probs_svm = classifier.predict_proba(emb.reshape(1, -1))[0]
                            pred_idx = int(np.argmax(probs_svm))
                            max_prob = float(probs_svm[pred_idx])
                            encoded_label = classifier.classes_[pred_idx]
                            pred_name = label_encoder.inverse_transform([encoded_label])[0]
                            is_known = max_prob >= RECOG_THRESHOLD

                            distance = None
                            if centroids is not None and pred_name in centroids:
                                distance = compute_embedding_distance(emb, centroids[pred_name])
                                is_known = is_known and (distance <= dist_threshold)

                            print(f"[DEBUG] svm probs: {probs_svm}, pred_idx: {pred_idx}, pred_label: {encoded_label}, pred_name: {pred_name}, max_prob: {max_prob:.3f}, distance: {distance}")

                            name = pred_name if is_known else "Unknown"
                            faces.append({"name": name, "bbox": (x1o, y1o, x2o, y2o), "prob": max_prob, "distance": distance})
                        except Exception as e:
                            print(f"[WARN] Classification failed: {e}")
                            faces.append({"name": "Unknown", "bbox": (x1o, y1o, x2o, y2o), "prob": 0.0, "distance": None})
                    print(f"[DEBUG] embeddings shape: {embeddings.shape}")


        # Draw and log
        for f in faces:
            name, prob, distance = f["name"], f["prob"], f["distance"]
            x1o, y1o, x2o, y2o = f["bbox"]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name}" if name == "Unknown" else f"{name} {prob:.2f}"
            cv2.rectangle(annotated_frame, (x1o, y1o), (x2o, y2o), color, 2)
            cv2.putText(annotated_frame, label, (x1o, max(0, y1o - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if name == "Unknown":
                unknown_in_frame = True
                ts_str = ts.strftime('%Y%m%d_%H%M%S%f')
                face_path = os.path.join(LOGS_UNKNOWN_DIR, f"unknown_{ts_str}.jpg")
                try:
                    crop = original_frame[max(0, y1o):min(orig_h, y2o), max(0, x1o):min(orig_w, x2o)]
                    if crop.size: cv2.imwrite(face_path, crop)
                    append_csv(os.path.join(LOGS_UNKNOWN_DIR, "detections.csv"),
                               ["timestamp", "frame", "file", "x1", "y1", "x2", "y2", "prob", "distance"],
                               [ts.isoformat(), frame_num, os.path.basename(face_path), x1o, y1o, x2o, y2o,
                                f"{prob:.3f}", "" if distance is None else f"{distance:.4f}"])
                except Exception as e:
                    print(f"[WARN] Failed saving unknown: {e}")
            else:
                known_names_in_frame.add(name)
                faces_by_name.setdefault(name, []).append(f)

        # Save unknown annotated frame
        if unknown_in_frame:
            ts_str = ts.strftime('%Y%m%d_%H%M%S%f')
            frame_path = os.path.join(ANNOTATED_UNKNOWN_DIR, f"frame_{frame_num}_{ts_str}.jpg")
            try: cv2.imwrite(frame_path, annotated_frame)
            except: pass

        # Save known periodically
        if known_names_in_frame:
            now = ts
            for name in known_names_in_frame:
                last = known_last_saved.get(name)
                if last is None or now - last >= timedelta(minutes=KNOWN_SAVE_INTERVAL_MIN):
                    known_last_saved[name] = now
                    for f in faces_by_name.get(name, []):
                        x1o, y1o, x2o, y2o = f["bbox"]
                        append_csv(os.path.join(LOGS_KNOWN_DIR, "detections.csv"),
                                   ["timestamp", "frame", "name", "x1", "y1", "x2", "y2", "prob", "distance"],
                                   [ts.isoformat(), frame_num, name, x1o, y1o, x2o, y2o,
                                    f"{f['prob']:.3f}", "" if f["distance"] is None else f"{f['distance']:.4f}"])
                    ts_str = ts.strftime('%Y%m%d_%H%M%S%f')
                    cv2.imwrite(os.path.join(ANNOTATED_KNOWN_DIR, f"frame_{frame_num}_{ts_str}.jpg"), annotated_frame)

        processed += 1

        try:
            display_q.put((frame_num, ts, annotated_frame), timeout=0.01)
        except:
            try:
                _ = display_q.get_nowait()
                display_q.put((frame_num, ts, annotated_frame), timeout=0.01)
            except: pass

    print(f"[INFO] Processed {processed} frames.")

# --- Main ---
cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)
frame_q, display_q, stop_event = Queue(CAPTURE_QUEUE_SIZE), Queue(DISPLAY_QUEUE_SIZE), Event()

print("[INFO] Starting face recognition...")
grab_t = Thread(target=grab_frames, args=(cap, frame_q, stop_event), daemon=True)
proc_t = Thread(target=process_frames, args=(frame_q, display_q, stop_event), daemon=True)
grab_t.start(), proc_t.start()

try:
    while not stop_event.is_set():
        try:
            frame_num, ts, annotated = display_q.get(timeout=0.05)
        except Empty:
            if not grab_t.is_alive() and not proc_t.is_alive(): break
            if cv2.waitKey(1) & 0xFF == ord("q"): stop_event.set()
            continue
        cv2.imshow("MTCNN + Face Recognition", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break
finally:
    stop_event.set()
    grab_t.join(timeout=1.0), proc_t.join(timeout=2.0)
    cv2.destroyAllWindows()
print(f"[INFO] Exiting.")
