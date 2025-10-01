import os
import sys
import json
import time
import argparse
from collections import deque
from typing import Dict, Tuple, List, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T

# Ensure folder is importable
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PARENT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if PARENT_ROOT not in sys.path:
    sys.path.insert(0, PARENT_ROOT)

# Import backbone (MobileNet + LSTM)
try:
    from LSTM_behavior_mobilenet import MobileNetLSTM  # type: ignore
except Exception as e:
    raise ImportError(f"Failed to import MobileNetLSTM: {e}")

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
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_classes(model_dir: str) -> List[str]:
    classes_path = os.path.join(model_dir, 'classes.json')
    if not os.path.isfile(classes_path):
        raise FileNotFoundError(f"Missing classes.json at {classes_path}")
    with open(classes_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    try:
        return [data[str(i)] for i in sorted(map(int, data.keys()))]
    except Exception:
        if isinstance(data, dict):
            return list(data.values())
        if isinstance(data, list):
            return data
        raise RuntimeError('classes.json format not recognized')

class SimpleTracker:
    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 10):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}

    @staticmethod
    def iou(a: np.ndarray, b: np.ndarray) -> float:
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        interW = max(0, xB - xA); interH = max(0, yB - yA)
        interArea = interW * interH
        areaA = max(0, (a[2]-a[0])) * max(0, (a[3]-a[1]))
        areaB = max(0, (b[2]-b[0])) * max(0, (b[3]-b[1]))
        union = areaA + areaB - interArea + 1e-6
        return float(interArea / union)

    def update(self, detections: List[np.ndarray]) -> Dict[int, np.ndarray]:
        assigned = set()
        id2box: Dict[int, np.ndarray] = {}
        for det in detections:
            best_iou, best_id = 0.0, None
            for tid, info in self.tracks.items():
                if tid in assigned:
                    continue
                iou_v = self.iou(info['bbox'], det)
                if iou_v > best_iou:
                    best_iou = iou_v
                    best_id = tid
            if best_id is not None and best_iou >= self.iou_thresh:
                self.tracks[best_id]['bbox'] = det
                self.tracks[best_id]['missed'] = 0
                assigned.add(best_id)
                id2box[best_id] = det
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {'bbox': det, 'missed': 0}
                assigned.add(tid)
                id2box[tid] = det
        # age others
        for tid in list(self.tracks.keys()):
            if tid not in assigned:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > self.max_missed:
                    del self.tracks[tid]
        return id2box

    def get_active_tracks(self) -> Dict[int, np.ndarray]:
        return {tid: info['bbox'] for tid, info in self.tracks.items()}

class BehaviorInferenceMobileNet:
    def __init__(
        self,
        model_dir: str = os.path.join('models', 'LSTM_behavior_mobilenet'),
        yolo_weights: str = None,
        conf_threshold: float = 0.6,
        device_str: str = None,
        log_path: str = os.path.join('logs', 'behavior_inference_mobilenet.csv'),
        sample_stride: int = 8,
    ):
        # Device
        self.device = torch.device(device_str) if device_str else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Config / classes
        self.cfg = load_config(model_dir)
        self.classes = load_classes(model_dir)
        self.seq_len = int(self.cfg.get('sequence_length', 32))
        self.img_size = int(self.cfg.get('img_size', 160))
        self.hidden_size = int(self.cfg.get('hidden_size', 512))
        self.num_layers = int(self.cfg.get('num_layers', 3))
        self.bidirectional = bool(self.cfg.get('bidirectional', False))
        self.dropout = float(self.cfg.get('dropout', 0.3))
        self.freeze_cnn = bool(self.cfg.get('freeze_cnn', True))

        # Build model
        self.model = MobileNetLSTM(
            num_classes=len(self.classes),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            freeze_cnn=self.freeze_cnn,
        ).to(self.device)
        ckpt_path = os.path.join(model_dir, 'best_model.pth')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state = checkpoint.get('model_state', checkpoint)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # Transforms
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((self.img_size, self.img_size))
        self.normalize = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

        # YOLO
        if YOLO is None:
            raise ImportError(f"Ultralytics not available: {_yolo_import_error}. Install with: pip install ultralytics")
        if yolo_weights is None:
            local_yolo = os.path.join('models', 'YOLOv8', 'yolov8n.pt')
            yolo_weights = local_yolo if os.path.isfile(local_yolo) else 'yolov8n.pt'
        self.yolo = YOLO(yolo_weights)
        self.yolo_conf = float(conf_threshold)

        # Tracker & buffers
        self.tracker = SimpleTracker(iou_thresh=0.3, max_missed=10)
        self.buffers: Dict[int, deque] = {}
        self.last_predictions: Dict[int, Tuple[str, float, float]] = {}
        self.last_logits: Dict[int, np.ndarray] = {}
        self.sample_stride = max(1, int(sample_stride))

        # Motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
        self.motion_pixel_threshold = 800
        self.motion_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

        # Neutral class inference
        self.neutral_labels = {c for c in self.classes if c.lower() == 'neutral'}
        if not self.neutral_labels and len(self.classes) > 0:
            self.neutral_labels = {self.classes[0]}

        # Logger
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        if not os.path.isfile(self.log_path):
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write('timestamp,person_id,label,confidence\n')

    @torch.no_grad()
    def preprocess_crop(self, frame_bgr: np.ndarray, box: np.ndarray) -> torch.Tensor:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(w-1, x1)); y1 = max(0, min(h-1, y1))
        x2 = max(0, min(w-1, x2)); y2 = max(0, min(h-1, y2))
        if x2 <= x1 or y2 <= y1:
            cx, cy = w//2, h//2
            x1, y1 = max(0, cx-10), max(0, cy-10)
            x2, y2 = min(w-1, cx+10), min(h-1, cy+10)
        crop = frame_bgr[y1:y2, x1:x2, :]
        if crop.size == 0:
            crop = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t = self.to_tensor(crop_rgb)
        t = self.resize(t)
        t = self.normalize(t)
        return t.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed_frame(self, frame_bgr: np.ndarray, box: np.ndarray) -> torch.Tensor:
        x = self.preprocess_crop(frame_bgr, box)  # (1,3,H,W)
        feats = self.model.cnn_features(x)
        feats = self.model.pool(feats)  # (1,1280,1,1)
        feats = feats.view(1, -1)  # (1,1280)
        return feats

    @torch.no_grad()
    def batch_embed(self, frame_bgr: np.ndarray, pid_boxes: Dict[int, np.ndarray]) -> Dict[int, torch.Tensor]:
        if not pid_boxes:
            return {}
        crops, pids = [], []
        h, w = frame_bgr.shape[:2]
        for pid, box in pid_boxes.items():
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, min(w-1, x1)); y1 = max(0, min(h-1, y1))
            x2 = max(0, min(w-1, x2)); y2 = max(0, min(h-1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2, :]
            if crop.size == 0: continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = self.to_tensor(crop_rgb)
            t = self.resize(t)
            t = self.normalize(t)
            crops.append(t); pids.append(pid)
        if not crops:
            return {}
        batch = torch.stack(crops, dim=0).to(self.device)
        feats = self.model.cnn_features(batch)
        feats = self.model.pool(feats)
        feats = feats.view(feats.size(0), -1)  # (B,1280)
        out: Dict[int, torch.Tensor] = {}
        for i, pid in enumerate(pids):
            out[pid] = feats[i:i+1]
        return out

    @torch.no_grad()
    def predict_sequence(self, emb_seq: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        # emb_seq: (1,T,F)
        lstm_out, _ = self.model.lstm(emb_seq)
        last = lstm_out[:, -1, :]
        logits = self.model.classifier(last)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        return int(idx.item()), float(conf.item()), logits.detach().cpu().numpy()[0]

    def log_prediction(self, pid: int, label: str, conf: float):
        if label in self.neutral_labels:
            return
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"{ts},{pid},{label},{conf:.4f}\n")

    def run(self, source: Union[int, str] = 0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {'camera index' if isinstance(source,int) else 'video file'}: {source}")
        print(f"Running MobileNet-LSTM inference on device: {self.device}")
        if isinstance(source, int):
            print(f"Using webcam index {source}")
        else:
            print(f"Processing video file: {source}")
        print("Press 'q' to quit window early.")

        try:
            prev_time = time.time()
            fps = 0.0
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print('Failed to read frame.')
                    break
                fgmask = self.bg_subtractor.apply(frame)
                _, fg_bin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
                fg_bin = cv2.dilate(fg_bin, self.motion_dilate_kernel, iterations=1)
                motion_pixels = cv2.countNonZero(fg_bin)
                motion_detected = motion_pixels > self.motion_pixel_threshold

                if motion_detected:
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
                                if clss[i] == 0:
                                    detections.append(xyxy[i])
                    id_to_box = self.tracker.update(detections)
                else:
                    id_to_box = self.tracker.get_active_tracks()

                embeddings_this_frame: Dict[int, torch.Tensor] = {}
                if motion_detected and (frame_idx % self.sample_stride == 0):
                    embeddings_this_frame = self.batch_embed(frame, id_to_box)

                for pid, box in id_to_box.items():
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    label_text = f"ID {pid}: neutral"
                    predicted_label = None; predicted_conf = None

                    if motion_detected:
                        if pid not in self.buffers:
                            self.buffers[pid] = deque(maxlen=self.seq_len)
                        if pid in embeddings_this_frame:
                            self.buffers[pid].append(embeddings_this_frame[pid].squeeze(0))
                        if len(self.buffers[pid]) == self.seq_len:
                            emb_seq = torch.stack(list(self.buffers[pid]), dim=0).unsqueeze(0).to(self.device)
                            idx, conf, logits = self.predict_sequence(emb_seq)
                            predicted_label = self.classes[idx] if 0 <= idx < len(self.classes) else str(idx)
                            predicted_conf = conf
                            self.last_logits[pid] = logits
                            if predicted_label in self.neutral_labels:
                                label_text = f"ID {pid}: neutral"
                            else:
                                label_text = f"ID {pid}: {predicted_label} ({conf*100:.1f}%)"
                            if predicted_label not in self.neutral_labels:
                                last = self.last_predictions.get(pid)
                                now = time.time(); should_log = False
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
                    # Text
                    y_text = max(10, y1-10)
                    cv2.putText(frame, label_text, (x1,y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

                # FPS overlay
                now_t = time.time()
                dt = now_t - prev_time
                if dt > 0:
                    fps = 0.9*fps + 0.1*(1.0/dt) if fps > 0 else (1.0/dt)
                prev_time = now_t
                cv2.putText(frame, f"FPS: {fps:.1f} | Motion: {'YES' if motion_detected else 'NO'}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255) if motion_detected else (200,200,200), 2, cv2.LINE_AA)

                cv2.imshow('Behavior Recognition (MobileNet)', frame)
                delay = 1
                if not isinstance(source, int):
                    vid_fps = cap.get(cv2.CAP_PROP_FPS)
                    if vid_fps and vid_fps > 0:
                        delay = max(1, int(1000/vid_fps))
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q'):
                    break
                frame_idx += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description='Real-time / File Behavior Recognition Inference (YOLO + MobileNetV2 + LSTM)')
    p.add_argument('--source', type=str, default='0', help='Webcam index or video path (default: 0)')
    p.add_argument('--camera', type=int, default=None, help='(Deprecated) explicit webcam index')
    p.add_argument('--yolo-weights', type=str, default=None, help='Path to YOLOv8 weights (default tries local yolov8n.pt)')
    p.add_argument('--conf', type=float, default=0.6, help='YOLO confidence threshold (default: 0.6)')
    p.add_argument('--device', type=str, default=None, help="Torch device string (e.g. 'cuda', 'cpu')")
    p.add_argument('--model-dir', type=str, default=os.path.join('models','LSTM_behavior_mobilenet'), help='Directory with best_model.pth + config + classes')
    p.add_argument('--log', type=str, default=os.path.join('logs','behavior_inference_mobilenet.csv'), help='CSV log path')
    p.add_argument('--stride', type=int, default=2, help='Frame sampling stride for embeddings (default: 2)')
    args = p.parse_args()
    if args.camera is not None:
        args.source = str(args.camera)
    if args.source.isdigit() and not os.path.isfile(args.source):
        try:
            args.source_val = int(args.source)
        except ValueError:
            args.source_val = args.source
    else:
        args.source_val = args.source
    return args

if __name__ == '__main__':
    args = parse_args()
    engine = BehaviorInferenceMobileNet(
        model_dir=args.model_dir,
        yolo_weights=args.yolo_weights,
        conf_threshold=args.conf,
        device_str=args.device,
        log_path=args.log,
        sample_stride=args.stride,
    )
    engine.run(source=args.source_val)
