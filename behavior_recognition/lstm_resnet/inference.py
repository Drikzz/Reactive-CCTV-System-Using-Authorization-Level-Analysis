import os
import sys
import json
import time
import argparse
import logging
from collections import deque
from typing import Dict, Tuple, List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the trained architecture definition
try:
    from LSTM_behavior_recognition import CNNLSTM  # type: ignore
except Exception as e:
    raise ImportError(
        f"Failed to import CNNLSTM from behavior_recognition.LSTM_behavior_recognition: {e}"
    )

# YOLO (Ultralytics)
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _yolo_import_error = e


def load_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing config.json at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_classes(model_dir: str) -> List[str]:
    classes_path = os.path.join(model_dir, "classes.json")
    if not os.path.isfile(classes_path):
        raise FileNotFoundError(f"Missing classes.json at {classes_path}")
    with open(classes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect keys as strings of ints
    try:
        ordered = [data[str(i)] for i in sorted(map(int, data.keys()))]
        return ordered
    except Exception:
        # Fallback: try preserving order of values
        if isinstance(data, dict):
            return list(data.values())
        if isinstance(data, list):
            return data
        raise RuntimeError("classes.json format not recognized")


class SimpleTracker:
    """Simple IoU-based tracker to assign stable IDs to detections across frames."""

    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 10):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}  # id -> {bbox, missed}

    @staticmethod
    def iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
        boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
        union = boxAArea + boxBArea - interArea + 1e-6
        return float(interArea / union)

    def update(self, detections: List[np.ndarray]) -> Dict[int, np.ndarray]:
        # detections: list of [x1,y1,x2,y2]
        assigned_tracks = set()
        id_to_box: Dict[int, np.ndarray] = {}

        # Greedy matching: for each detection, find best available track
        for det in detections:
            best_iou = 0.0
            best_id = None
            for tid, info in self.tracks.items():
                if tid in assigned_tracks:
                    continue
                iou_val = self.iou(info["bbox"], det)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_id = tid
            if best_id is not None and best_iou >= self.iou_thresh:
                # Match existing track
                self.tracks[best_id]["bbox"] = det
                self.tracks[best_id]["missed"] = 0
                assigned_tracks.add(best_id)
                id_to_box[best_id] = det
            else:
                # Create new track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": det, "missed": 0}
                assigned_tracks.add(tid)
                id_to_box[tid] = det

        # Increment missed count for unmatched tracks
        for tid in list(self.tracks.keys()):
            if tid not in assigned_tracks:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]

        return id_to_box

    def get_active_tracks(self) -> Dict[int, np.ndarray]:
        """Return current active tracks without updating (used when skipping detection)."""
        return {tid: info["bbox"] for tid, info in self.tracks.items()}


class BehaviorInference:
    def __init__(
        self,
        model_dir: str = os.path.join("models", "LSTM_behavior_recognition"),
        yolo_weights: str = None,
        conf_threshold: float = 0.65,
        device_str: str = None,
        log_path: str = os.path.join("logs", "behavior_inference.csv"),
        sample_stride: int = 3,
    ):
        # Device
        if device_str:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config and classes
        self.cfg = load_config(model_dir)
        self.classes = load_classes(model_dir)
        self.seq_len: int = int(self.cfg.get("sequence_length", 32))
        self.img_size: int = int(self.cfg.get("img_size", 224))
        self.hidden_size: int = int(self.cfg.get("hidden_size", 512))
        self.num_layers: int = int(self.cfg.get("num_layers", 3))
        self.bidirectional: bool = bool(self.cfg.get("bidirectional", False))
        self.dropout: float = float(self.cfg.get("dropout", 0.3))
        self.freeze_cnn: bool = bool(self.cfg.get("freeze_cnn", True))

        # Build model and load weights
        self.model = CNNLSTM(
            num_classes=len(self.classes),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            freeze_cnn=self.freeze_cnn,
        ).to(self.device)
        ckpt_path = os.path.join(model_dir, "best_model.pth")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # Preprocessing transforms
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((self.img_size, self.img_size))
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # YOLO model
        if YOLO is None:
            raise ImportError(
                f"Ultralytics not available: {_yolo_import_error}. Install with: pip install ultralytics"
            )
        if yolo_weights is None:
            # Prefer local lightweight weights if available
            local_yolo = os.path.join("models", "YOLOv8", "yolov8n.pt")
            yolo_weights = local_yolo if os.path.isfile(local_yolo) else "yolov8n.pt"
        self.yolo = YOLO(yolo_weights)
        self.yolo_conf = float(conf_threshold)

        # Tracker and buffers
        self.tracker = SimpleTracker(iou_thresh=0.3, max_missed=10)
        self.buffers: Dict[int, deque] = {}
        self.last_predictions: Dict[int, Tuple[str, float, float]] = {}  # id -> (label, conf, timestamp)
        self.last_logits: Dict[int, np.ndarray] = {}  # id -> raw logits (numpy)

        # Frame sampling stride for feature extraction (embeddings); detection still per-frame when motion present
        self.sample_stride = max(1, int(sample_stride))

        # Motion detection (background subtractor)
        # Using MOG2 for robustness to lighting changes; tweak thresholds as needed.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
        self.motion_pixel_threshold = 800  # Minimum foreground pixels to consider motion present
        self.motion_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Infer neutral class name (case-insensitive match)
        self.neutral_labels = {c for c in self.classes if c.lower() == "neutral"}
        if not self.neutral_labels:
            # Fallback: treat index 0 as neutral if explicit neutral not found
            self.neutral_labels = {self.classes[0]}

        # Logger
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        if not os.path.isfile(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("timestamp,person_id,label,confidence\n")

    @torch.no_grad()
    def preprocess_crop(self, frame_bgr: np.ndarray, box: np.ndarray) -> torch.Tensor:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            # Fallback to tiny center crop
            cx, cy = w // 2, h // 2
            x1, y1 = max(0, cx - 10), max(0, cy - 10)
            x2, y2 = min(w - 1, cx + 10), min(h - 1, cy + 10)
        crop = frame_bgr[y1:y2, x1:x2, :]
        if crop.size == 0:
            crop = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        # BGR -> RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.to_tensor(crop_rgb)  # [0,1]
        tensor = self.resize(tensor)
        tensor = self.normalize(tensor)
        return tensor.unsqueeze(0).to(self.device)  # (1,3,H,W)

    @torch.no_grad()
    def embed_frame(self, frame_bgr: np.ndarray, box: np.ndarray) -> torch.Tensor:
        x = self.preprocess_crop(frame_bgr, box)  # (1,3,H,W)
        feats = self.model.cnn(x)  # (1,512,1,1)
        feats = feats.view(1, -1)  # (1,512)
        return feats  # device tensor

    @torch.no_grad()
    def batch_embed(self, frame_bgr: np.ndarray, pid_boxes: Dict[int, np.ndarray]) -> Dict[int, torch.Tensor]:
        """Batch embed multiple person crops for efficiency.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Current frame in BGR format.
        pid_boxes : Dict[int, np.ndarray]
            Mapping of track id -> box (x1,y1,x2,y2).

        Returns
        -------
        Dict[int, torch.Tensor]
            Mapping of track id -> embedding tensor (1,512).
        """
        if not pid_boxes:
            return {}
        crops = []
        pids = []
        h, w = frame_bgr.shape[:2]
        for pid, box in pid_boxes.items():
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2, :]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = self.to_tensor(crop_rgb)
            t = self.resize(t)
            t = self.normalize(t)
            crops.append(t)
            pids.append(pid)
        if not crops:
            return {}
        batch = torch.stack(crops, dim=0).to(self.device)  # (B,3,H,W)
        feats = self.model.cnn(batch)  # (B,512,1,1)
        feats = feats.view(feats.size(0), -1)  # (B,512)
        out: Dict[int, torch.Tensor] = {}
        for i, pid in enumerate(pids):
            out[pid] = feats[i : i + 1]  # keep batch dim (1,512)
        return out

    @torch.no_grad()
    def predict_sequence(self, emb_seq: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        # emb_seq: (1,T,512)
        lstm_out, _ = self.model.lstm(emb_seq)
        # Use last timestep representation (standard) for classification
        last = lstm_out[:, -1, :]
        logits = self.model.classifier(last)  # (1,num_classes)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        # Store logits (caller can fetch from self.last_logits)
        return int(idx.item()), float(conf.item()), logits.detach().cpu().numpy()[0]

    def log_prediction(self, person_id: int, label: str, conf: float):
        # Skip logging neutral behaviors entirely
        if label in self.neutral_labels:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{ts},{person_id},{label},{conf:.4f}\n")

    def run(self, source: Union[int, str] = 0):
        """Run real-time / file-based behavior inference.

        Parameters
        ----------
        source : int | str
            - int: webcam index (e.g., 0)
            - str: path to a video file
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open {'camera index' if isinstance(source, int) else 'video file'}: {source}"
            )

        print(f"Running inference on device: {self.device}")
        print(self.device)
        print(next(self.model.parameters()).device)
        print(self.yolo.device)  # or self.yolo.predict(...).results[0].speed

        if isinstance(source, int):
            print(f"Using webcam index {source}")
        else:
            print(f"Processing video file: {source}")
        print("Press 'q' to quit window early.")

        try:
            prev_time = time.time()
            fps = 0.0
            frame_idx = 0  # for sampling stride
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera.")
                    break
                # -------------------- MOTION DETECTION --------------------
                fgmask = self.bg_subtractor.apply(frame)
                # Reduce noise & shadows (shadows often ~127)
                _, fg_bin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
                fg_bin = cv2.dilate(fg_bin, self.motion_dilate_kernel, iterations=1)
                motion_pixels = cv2.countNonZero(fg_bin)
                motion_detected = motion_pixels > self.motion_pixel_threshold

                # -------------------- PERSON DETECTION & TRACKING --------------------
                if motion_detected:
                    # Run YOLO only when motion present
                    yolo_results = self.yolo.predict(
                        source=frame,
                        verbose=False,
                        conf=self.yolo_conf,
                        device=0 if self.device.type == 'cuda' else 'cpu'
                    )
                    detections: List[np.ndarray] = []
                    if yolo_results and len(yolo_results) > 0:
                        res = yolo_results[0]
                        boxes = res.boxes
                        if boxes is not None and boxes.xyxy is not None:
                            xyxy = boxes.xyxy.detach().cpu().numpy()
                            clss = boxes.cls.detach().cpu().numpy().astype(int)
                            for i in range(xyxy.shape[0]):
                                if clss[i] == 0:  # person class
                                    detections.append(xyxy[i])
                    id_to_box = self.tracker.update(detections)
                else:
                    # No motion: keep existing tracked boxes
                    id_to_box = self.tracker.get_active_tracks()

                # -------------------- BEHAVIOR CLASSIFICATION --------------------
                # Prepare batch embeddings only on sampled frames (stride) & when motion detected
                embeddings_this_frame: Dict[int, torch.Tensor] = {}
                if motion_detected and (frame_idx % self.sample_stride == 0):
                    embeddings_this_frame = self.batch_embed(frame, id_to_box)

                for pid, box in id_to_box.items():
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    predicted_label = None
                    predicted_conf = None
                    label_text = f"ID {pid}: neutral"

                    if motion_detected:
                        # Initialize buffer if new
                        if pid not in self.buffers:
                            self.buffers[pid] = deque(maxlen=self.seq_len)

                        # Only append embedding on sampled frames when we computed batch embeddings
                        if pid in embeddings_this_frame:
                            self.buffers[pid].append(embeddings_this_frame[pid].squeeze(0))

                        if len(self.buffers[pid]) == self.seq_len:
                            emb_seq = torch.stack(list(self.buffers[pid]), dim=0).unsqueeze(0).to(self.device)
                            idx, conf, logits = self.predict_sequence(emb_seq)
                            predicted_label = self.classes[idx] if 0 <= idx < len(self.classes) else str(idx)
                            predicted_conf = conf
                            self.last_logits[pid] = logits  # store raw logits

                            if predicted_label in self.neutral_labels:
                                label_text = f"ID {pid}: neutral"
                            else:
                                label_text = f"ID {pid}: {predicted_label} ({conf*100:.1f}%)"

                            # Logging logic: only for suspicious (non-neutral)
                            if predicted_label not in self.neutral_labels:
                                last = self.last_predictions.get(pid)
                                now = time.time()
                                should_log = False
                                if last is None:
                                    should_log = True
                                else:
                                    last_label, last_conf, _ = last
                                    if last_label != predicted_label or abs(last_conf - conf) > 0.05:
                                        should_log = True
                                if should_log:
                                    self.log_prediction(pid, predicted_label, conf)
                                    self.last_predictions[pid] = (predicted_label, conf, now)
                            else:
                                self.last_predictions[pid] = (predicted_label, predicted_conf or 0.0, time.time())
                    else:
                        # No motion: display neutral only (don't modify buffers to avoid stale sequences)
                        label_text = f"ID {pid}: neutral"

                    # Draw label text
                    y_text = max(10, y1 - 10)
                    cv2.putText(
                        frame,
                        label_text,
                        (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                # -------------------- FPS DISPLAY --------------------
                now_t = time.time()
                dt = now_t - prev_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
                prev_time = now_t
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f} | Motion: {'YES' if motion_detected else 'NO'}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255) if motion_detected else (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Behavior Recognition", frame)
                # For files, slow down to (approx) real-time if FPS known; else minimal delay
                delay = 1
                if not isinstance(source, int):
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps and fps > 0:
                        delay = max(1, int(1000 / fps))  # milliseconds
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q'):
                    break
                frame_idx += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Real-time / File Behavior Recognition Inference (YOLO + ResNet18 + LSTM)")
    p.add_argument(
        "--source",
        type=str,
        default="0",
        help="Webcam index (e.g. 0) or path to video file. Default: 0",
    )
    # Backwards compatibility: keep --camera (overrides --source if provided explicitly)
    p.add_argument(
        "--camera",
        type=int,
        default=None,
        help="(Deprecated) Webcam index; use --source instead.",
    )
    p.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="Path to YOLOv8 weights (default: models/YOLOv8/yolov8n.pt or auto-download)",
    )
    p.add_argument(
        "--conf", type=float, default=0.65, help="YOLO confidence threshold (default: 0.65)"
    )
    p.add_argument(
        "--device", type=str, default=None, help="Torch device string (e.g., 'cuda', 'cpu'). Default auto."
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join("models", "LSTM_behavior_recognition"),
        help="Directory containing best_model.pth, config.json, classes.json",
    )
    p.add_argument(
        "--log",
        type=str,
        default=os.path.join("logs", "behavior_inference.csv"),
        help="Path to CSV log file",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Frame sampling stride for embeddings (default: 3)",
    )
    args = p.parse_args()
    # Resolve source (camera index or file)
    if args.camera is not None:
        args.source = str(args.camera)
    # If source is digits and file doesn't exist -> treat as int
    if args.source.isdigit() and not os.path.isfile(args.source):
        try:
            args.source_val = int(args.source)
        except ValueError:
            args.source_val = args.source  # fallback
    else:
        args.source_val = args.source  # file path
    return args


if __name__ == "__main__":
    args = parse_args()
    engine = BehaviorInference(
        model_dir=args.model_dir,
        yolo_weights=args.yolo_weights,
        conf_threshold=args.conf,
        device_str=args.device,
        log_path=args.log,
        sample_stride=args.stride,
    )
    engine.run(source=args.source_val)
