import argparse
import importlib.util
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict, deque
import os

# Resolve repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class CombinedYOLOFaceBehavior:
    """
    Combine:
     - YOLOv8 + ByteTrack -> detection + tracking
     - MobileNetV3-Small -> behavior classification (per-track caching + smoothing)
     - FaceNet (from facenet_main.py) -> face recognition per track
    """
    def __init__(self, yolo_path, mobilenet_path, facenet_main_path, device=None, tracker_cfg="bytetrack.yaml",
                 reclass_interval=10, smooth_window=7, min_class_conf=0.6, recog_interval=5,
                 iou_update_threshold=0.3, centroid_update_px=80, identity_lock_frames=30, identity_lock_conf=0.90):
        # Load YOLO+tracker/mobilenet module (the tracker class file)
        yolo_mnv_path = Path(REPO_ROOT) / "behavior_recognition" / "mobilenetv3-small" / "yolo_mobilenet_tracker_mnv3small.py"
        if Path(yolo_path).exists():
            yolo_model_path = str(yolo_path)
        else:
            yolo_model_path = str(Path(yolo_path))  # leave as-is, YOLO will error if missing

        # Load the tracker module to reuse its MobileNet loader and transforms
        tracker_mod = load_module_from_path("yolo_mnv_tracker_mod", yolo_mnv_path)
        TrackerClass = getattr(tracker_mod, "YOLOMobileNetV3SmallTracker")

        # Instantiate tracker (it will load YOLO and MobileNet)
        # ensure we prefer a lightweight model if the provided path doesn't exist
        yolo_model_path = yolo_model_path
        # If the path doesn't exist, pass the hub model name so ultralytics will download it:
        if not Path(yolo_model_path).exists():
            print(f"[INFO] YOLO model not found at '{yolo_model_path}'. Falling back to hub model 'yolov8n.pt' (will download automatically).")
            yolo_model_path = "yolov8n.pt"
 
        self.tracker = TrackerClass(
            yolo_model_path=yolo_model_path,
            mobilenet_model_path=str(mobilenet_path),
            class_names=None,
            tracker=tracker_cfg,
            device=device,
            reclass_interval=reclass_interval,
            smoothing_window=smooth_window
        )
        self.tracker.min_class_conf = float(min_class_conf)

        # Load facenet_main module (for recognize_face_in_crop)
        facenet_path = Path(facenet_main_path)
        facenet_mod = load_module_from_path("facenet_main_mod", facenet_path)
        # prefer using recognize_face_in_crop if available
        self.recognize_face_fn = getattr(facenet_mod, "recognize_face_in_crop", None)
        if self.recognize_face_fn is None:
            raise RuntimeError("facenet_main module does not expose recognize_face_in_crop()")

        # Also bring in helper config if needed
        self.display_title = "YOLO + MobileNetV3-Small + FaceNet (Combined)"

        # add facenet track state (use same sizes as facenet_main expects)
        self.track_identities = {}
        self.track_face_history = defaultdict(lambda: deque(maxlen=300))  # EXTENDED_MEMORY_FRAMES in facenet_main
        self.track_body_history = defaultdict(lambda: deque(maxlen=60))
        self.track_last_face_frame = {}

        # load facenet module object so we can call its helpers
        self.facenet_mod = facenet_mod

        # NEW: control how often to run FaceNet per track
        self.recog_interval = int(recog_interval)  # frames between FaceNet runs per track
        self.track_last_recog_frame = {}           # track_id -> last frame idx where FaceNet ran
        self.identity_cache = {}                   # track_id -> (name, conf)

        # store last bbox per track to check continuity and prevent swaps
        self.track_last_bbox = {}          # track_id -> (x1,y1,x2,y2)
        # lock confident identities for a number of frames to avoid rapid switching
        self.track_identity_lock = {}      # track_id -> {"name": str, "frames_left": int}
        self.iou_update_threshold = float(iou_update_threshold)
        self.centroid_update_px = float(centroid_update_px)
        self.identity_lock_frames = int(identity_lock_frames)
        self.identity_lock_conf = float(identity_lock_conf)

    def process_video(self, video_source, display=True, save_output=None, conf_threshold=0.5, iou_threshold=0.7):
        # Open video
        if video_source == "webcam":
            cap = cv2.VideoCapture(0)
            is_webcam = True
        else:
            cap = cv2.VideoCapture(str(video_source))
            is_webcam = False

        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")

        fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        out_writer = None
        if save_output:
            out_path = Path(save_output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(str(out_path), fourcc, max(1, fps), (width, height))

        frame_idx = 0
        fps_window = []
        print(f"Processing {video_source} ({width}x{height} @ {fps}fps)")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                t0 = time.time()

                # Run YOLOv8 tracking (uses tracker.yolo inside class)
                results = self.tracker.yolo.track(
                    frame,
                    persist=True,
                    tracker=self.tracker.tracker_name,
                    classes=[0],
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )

                tracks_data = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    if boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                        bboxes = boxes.xyxy.cpu().numpy().astype(int)
                        for bbox, track_id in zip(bboxes, track_ids):
                            x1, y1, x2, y2 = bbox
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width, x2), min(height, y2)
                            if x2 <= x1 or y2 <= y1:
                                continue

                            person_crop = frame[y1:y2, x1:x2]

                            # Behavior classification (reuse tracker logic: reclass interval + smoothing)
                            do_classify = self.tracker._should_reclassify(track_id, frame_idx)
                            if do_classify:
                                class_res = self.tracker._classify_crop(person_crop)
                                # temporal smoothing
                                self.tracker.prob_history[track_id].append(class_res['probs'])
                                hist = self.tracker.prob_history[track_id]
                                if len(hist) == 1:
                                    smoothed = hist[0]
                                else:
                                    smoothed = np.mean(np.stack(hist, axis=0), axis=0)
                                cid = int(np.argmax(smoothed))
                                conf = float(smoothed[cid])
                                if conf < self.tracker.min_class_conf:
                                    behavior_name = "Neutral"
                                    behavior_conf = conf
                                else:
                                    behavior_name = self.tracker.class_names[cid]
                                    behavior_conf = conf
                                self.tracker.classification_cache[track_id] = {
                                    "class_name": behavior_name,
                                    "confidence": behavior_conf,
                                    "last_frame": frame_idx
                                }
                            else:
                                cached = self.tracker.classification_cache.get(track_id, {"class_name":"Neutral","confidence":0.0})
                                behavior_name = cached["class_name"]
                                behavior_conf = cached["confidence"]

                            # FaceNet recognition: only every self.recog_interval frames per track
                            # run FaceNet if interval reached (or forced by DEBUG_FORCE_RECOG)
                            do_recog = (frame_idx - self.track_last_recog_frame.get(int(track_id), -999)) >= self.recog_interval \
                                       or os.getenv("DEBUG_FORCE_RECOG") == "1"

                            # skip very small crops (likely not usable by MTCNN)
                            h, w = person_crop.shape[:2]
                            if h < 40 or w < 40:
                                face_result = {"name": "Unknown", "confidence": 0.0}
                            elif do_recog:
                                # call FaceNet with original crop + full frame + bbox (as facenet_main expects)
                                face_result = self.recognize_face_fn(person_crop, frame, (x1, y1, x2, y2))
                                name = face_result.get("name", "Unknown")
                                conf = float(face_result.get("confidence", 0.0) or 0.0)

                                # update cache / last run
                                self.identity_cache[int(track_id)] = (name, conf)
                                self.track_last_recog_frame[int(track_id)] = frame_idx

                                # debug: dump full face_result and save crop for inspection
                                if os.getenv("DEBUG_FACENET") == "1":
                                    print(f"[FACENET] track={track_id} frame={frame_idx} -> {face_result}")
                                    try:
                                        dbg_dir = Path("debug_facenet_crops")
                                        dbg_dir.mkdir(exist_ok=True)
                                        crop_path = dbg_dir / f"{Path(video_source).stem}_f{frame_idx}_t{track_id}.jpg"
                                        cv2.imwrite(str(crop_path), person_crop)
                                        print("[FACENET] saved crop:", crop_path)
                                    except Exception as e:
                                        print("[FACENET] failed saving crop:", e)
                            else:
                                # use cached identity if available
                                name, conf = self.identity_cache.get(int(track_id), ("Unknown", 0.0))
                                face_result = {"name": name, "confidence": conf}

                            # bounding-box continuity checks to avoid swapping
                            last_bbox = self.track_last_bbox.get(int(track_id))
                            current_bbox = (x1, y1, x2, y2)
                            iou_val = self._iou(last_bbox, current_bbox) if last_bbox is not None else 1.0
                            last_cent = self._centroid(last_bbox) if last_bbox is not None else None
                            curr_cent = self._centroid(current_bbox)
                            centroid_dist = ( (last_cent[0]-curr_cent[0])**2 + (last_cent[1]-curr_cent[1])**2 )**0.5 if last_cent is not None else 0.0

                            # check identity lock first
                            locked = self.track_identity_lock.get(int(track_id))
                            if locked and locked.get("frames_left",0) > 0:
                                identity_name = locked["name"]
                                identity_conf = self.identity_cache.get(int(track_id), ("Unknown",0.0))[1]
                                # decrement lock counter
                                locked["frames_left"] -= 1
                                # still update last bbox to keep continuity
                                self.track_last_bbox[int(track_id)] = current_bbox
                                tracks_data.append({
                                    "track_id": int(track_id),
                                    "bbox": current_bbox,
                                    "behavior": behavior_name,
                                    "behavior_conf": behavior_conf,
                                    "identity": identity_name,
                                    "identity_conf": identity_conf
                                })
                                continue

                            # If last bbox exists and movement is large AND IoU is low, skip face recognition/update
                            if last_bbox is not None and (iou_val < self.iou_update_threshold and centroid_dist > self.centroid_update_px):
                                # treat as potential swap / occlusion - keep cached identity if available
                                name, conf = self.identity_cache.get(int(track_id), ("Unknown", 0.0))
                                face_result = {"name": name, "confidence": conf}
                            else:
                                # proceed with normal behavior: maybe classify & run FaceNet (cached or fresh)
                                do_recog = (frame_idx - self.track_last_recog_frame.get(int(track_id), -999)) >= self.recog_interval

                                if do_recog:
                                    face_result = self.recognize_face_fn(person_crop, frame, current_bbox)
                                    name = face_result.get("name", "Unknown")
                                    conf = float(face_result.get("confidence", 0.0) or 0.0)
                                    # update cache
                                    self.identity_cache[int(track_id)] = (name, conf)
                                    self.track_last_recog_frame[int(track_id)] = frame_idx

                                    # if model is highly confident, lock identity for a few frames
                                    if conf >= self.identity_lock_conf and name != "Unknown":
                                        self.track_identity_lock[int(track_id)] = {"name": name, "frames_left": self.identity_lock_frames}
                                else:
                                    name, conf = self.identity_cache.get(int(track_id), ("Unknown", 0.0))
                                    face_result = {"name": name, "confidence": conf}

                                # update continuity record
                                self.track_last_bbox[int(track_id)] = current_bbox

                            # existing facenet_main state update (unchanged)
                            try:
                                self.facenet_mod.update_track_identity(
                                    int(track_id), face_result, person_crop,
                                    self.track_identities, self.track_face_history,
                                    self.track_body_history, frame_idx
                                )
                                frames_since_face = frame_idx - self.track_last_face_frame.get(int(track_id), frame_idx)
                                identity_name, identity_conf = self.facenet_mod.get_consensus_identity(
                                    int(track_id), self.track_identities, self.track_face_history, frames_since_face
                                )
                                if face_result.get("name") and face_result["name"] != "Unknown":
                                    self.track_last_face_frame[int(track_id)] = frame_idx
                            except Exception:
                                identity_name = face_result.get("name","Unknown")
                                identity_conf = float(face_result.get("confidence",0.0) or 0.0)

                            tracks_data.append({
                                "track_id": int(track_id),
                                "bbox": (x1, y1, x2, y2),
                                "behavior": behavior_name,
                                "behavior_conf": behavior_conf,
                                "identity": identity_name,
                                "identity_conf": identity_conf
                            })

                # annotate
                annotated = frame.copy()
                for td in tracks_data:
                    x1, y1, x2, y2 = td["bbox"]
                    tid = td["track_id"]
                    behavior = td["behavior"]
                    bconf = td["behavior_conf"]
                    ident = td["identity"]
                    iconf = td["identity_conf"]

                    color = (0, 200, 0) if ident != "Unknown" else (0, 0, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{tid} {ident} ({iconf:.2f}) | {behavior} ({bconf:.2f})"
                    cv2.putText(annotated, label, (x1+2, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # display / save
                if display:
                    cv2.imshow(self.display_title, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                if out_writer:
                    out_writer.write(annotated)

                # perf
                dt = time.time() - t0
                fps_window.append(dt)
                if len(fps_window) > 30:
                    fps_window.pop(0)

        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            if display:
                cv2.destroyAllWindows()

    # helper utils
    @staticmethod
    def _iou(boxA, boxB):
        # box: (x1,y1,x2,y2)
        if boxA is None or boxB is None:
            return 0.0
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
        boxBArea = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
        denom = float(boxAArea + boxBArea - interArea)
        return (interArea / denom) if denom > 0 else 0.0

    @staticmethod
    def _centroid(box):
        x1,y1,x2,y2 = box
        return ((x1+x2)/2.0, (y1+y2)/2.0)

def main():
    parser = argparse.ArgumentParser(description="Combined YOLOv8 + MobileNetV3-Small + FaceNet pipeline")
    parser.add_argument("--video", type=str, help="Path to video file (or 'webcam')", required=True)
    parser.add_argument("--yolo-model", type=str, default="models/YOLOv8/yolov8n.pt")
    parser.add_argument("--mobilenet-model", type=str, default="models/mobilenetv3-small/mobilenet_v3_small_transfer.pth")
    parser.add_argument("--facenet-main", type=str, default=str(REPO_ROOT / "face_recognition" / "Facenet" / "facenet_main.py"))
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save", type=str, help="Save annotated video to path")
    parser.add_argument("--device", type=str, choices=["cuda","cpu"], help="Device to use")
    parser.add_argument("--min-class-conf", type=float, default=0.6)
    args = parser.parse_args()

    video_src = "webcam" if args.video == "webcam" else args.video

    comb = CombinedYOLOFaceBehavior(
        yolo_path=args.yolo_model,
        mobilenet_path=args.mobilenet_model,
        facenet_main_path=args.facenet_main,
        device=args.device,
        min_class_conf=args.min_class_conf
    )
    comb.process_video(video_src, display=not args.no_display, save_output=args.save)

if __name__ == "__main__":
    main()