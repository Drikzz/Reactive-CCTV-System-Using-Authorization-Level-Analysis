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


class BehaviorInference:
    def __init__(
        self,
        model_dir: str = os.path.join("models", "LSTM_behavior_recognition"),
        yolo_weights: str = None,
        conf_threshold: float = 0.4,
        device_str: str = None,
        log_path: str = os.path.join("logs", "behavior_inference.csv"),
    ):
        # Device
        if device_str:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config and classes
        self.cfg = load_config(model_dir)
        self.classes = load_classes(model_dir)
        self.seq_len: int = int(self.cfg.get("sequence_length", 8))
        self.img_size: int = int(self.cfg.get("img_size", 224))
        self.hidden_size: int = int(self.cfg.get("hidden_size", 256))
        self.num_layers: int = int(self.cfg.get("num_layers", 2))
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
    def predict_sequence(self, emb_seq: torch.Tensor) -> Tuple[int, float]:
        # emb_seq: (1,T,512)
        lstm_out, _ = self.model.lstm(emb_seq)
        last = lstm_out[:, -1, :]
        logits = self.model.classifier(last)
        probs = torch.softmax(logits, dim=1)
        print(probs)
        conf, idx = torch.max(probs, dim=1)
        return int(idx.item()), float(conf.item())

    def log_prediction(self, person_id: int, label: str, conf: float):
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
        if isinstance(source, int):
            print(f"Using webcam index {source}")
        else:
            print(f"Processing video file: {source}")
        print("Press 'q' to quit window early.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera.")
                    break

                # YOLO person detection
                yolo_results = self.yolo.predict(source=frame, verbose=False, conf=self.yolo_conf, device=0 if self.device.type == 'cuda' else 'cpu')
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

                # Update tracker to get IDs
                id_to_box = self.tracker.update(detections)

                # For each tracked person, update embedding buffer and possibly predict
                for pid, box in id_to_box.items():
                    # Draw bbox
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"ID {pid}: collecting..."

                    # Update buffer
                    if pid not in self.buffers:
                        self.buffers[pid] = deque(maxlen=self.seq_len)

                    emb = self.embed_frame(frame, box)  # (1,512)
                    self.buffers[pid].append(emb.squeeze(0))  # (512,)

                    if len(self.buffers[pid]) == self.seq_len:
                        # Build (1,T,512)
                        emb_seq = torch.stack(list(self.buffers[pid]), dim=0).unsqueeze(0).to(self.device)
                        idx, conf = self.predict_sequence(emb_seq)
                        label = self.classes[idx] if 0 <= idx < len(self.classes) else str(idx)
                        label_text = f"ID {pid}: {label} ({conf*100:.1f}%)"

                        # Log predictions (only if new or changed)
                        last = self.last_predictions.get(pid)
                        now = time.time()
                        if last is None or last[0] != label or (now - last[2]) > 2.0:
                            self.log_prediction(pid, label, conf)
                            self.last_predictions[pid] = (label, conf, now)

                    # Put label text
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
        "--conf", type=float, default=0.4, help="YOLO confidence threshold (default: 0.4)"
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
    )
    engine.run(source=args.source_val)
