"""Face-only combined pipeline (YOLO + ByteTrack + FaceNet).

This is a Streamlit-friendly replacement for `combined_yolo_facenet_mnv3.py`.

Goals:
- Keep the *tracking stability* benefits of YOLOv8 + ByteTrack (persistent track IDs)
- Keep FaceNet recognition via a pluggable `facenet_main.py` path (same dynamic loading pattern)
- REMOVE behavior recognition entirely (no MobileNet, no yolo_mobilenet_tracker)

The class is designed to be called from a background processing thread.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class DetectionRow:
    track_id: int
    identity: str
    authorization: str
    identity_conf: float
    camera: str
    timestamp: str
    bbox: Tuple[int, int, int, int]


class CombinedYOLOFaceOnly:
    """YOLOv8 + ByteTrack tracking + FaceNet recognition (face-only)."""

    def __init__(
        self,
        yolo_model_path: str | Path,
        facenet_main_path: str | Path,
        *,
        authorization_map: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.7,
        imgsz: int = 640,
        resize_factor: float = 1.0,
        frame_skip: int = 1,
        recog_interval: int = 10,
        tracker_cfg: str = "bytetrack.yaml",
    ) -> None:
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.resize_factor = float(resize_factor)
        self.frame_skip = max(int(frame_skip), 1)
        self.recog_interval = max(int(recog_interval), 1)
        self.tracker_cfg = str(tracker_cfg)
        self.device = device

        self.authorization_map = authorization_map or {}

        # --- Load FaceNet module (dynamic path like old pipeline)
        self.facenet_path = Path(facenet_main_path)
        if not self.facenet_path.is_absolute():
            self.facenet_path = (REPO_ROOT / self.facenet_path).resolve()
        if not self.facenet_path.exists():
            raise FileNotFoundError(f"FaceNet main not found: {self.facenet_path}")

        self.facenet_mod = load_module_from_path("facenet_main_faceonly", self.facenet_path)
        self.recognize_face_fn = getattr(self.facenet_mod, "recognize_face_in_crop", None)
        if self.recognize_face_fn is None:
            raise AttributeError(
                f"{self.facenet_path.name} does not provide recognize_face_in_crop(person_crop, original_frame, person_bbox)"
            )

        # --- Load YOLO
        yolo_path = Path(yolo_model_path)
        if not yolo_path.is_absolute():
            yolo_path = (REPO_ROOT / yolo_path).resolve()
        if not yolo_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")

        # Ultralytics YOLO class is imported inside FaceNet modules in your repo, but
        # we avoid coupling to that by importing directly here.
        from ultralytics import YOLO  # type: ignore

        self.yolo = YOLO(str(yolo_path))

        # Per-track recognition throttle
        self._last_recog_frame: Dict[int, int] = {}
        self._identity_cache: Dict[int, Tuple[str, float]] = {}

        self._frame_idx = 0

    def get_authorization_level(self, identity_name: str) -> str:
        if not identity_name or identity_name == "Unknown":
            return "Unauthorized"
        return self.authorization_map.get(identity_name.lower(), "Partially Authorized")

    @staticmethod
    def get_authorization_color(auth_level: str):
        color_map = {
            "Authorized": (0, 255, 0),
            "Partially Authorized": (0, 165, 255),
            "Unauthorized": (0, 0, 255),
        }
        return color_map.get(auth_level, (128, 128, 128))

    def _should_recognize(self, track_id: int) -> bool:
        last = self._last_recog_frame.get(track_id, -10_000)
        return (self._frame_idx - last) >= self.recog_interval

    def process_frame(self, frame_bgr) -> Tuple[Any, List[Dict[str, Any]]]:
        """Process a single frame.

        Returns:
            annotated_frame_bgr, detections (list[dict])
        """
        self._frame_idx += 1
        if self._frame_idx % self.frame_skip != 0:
            return frame_bgr, []

        orig = frame_bgr

        # Resize for YOLO if requested
        if self.resize_factor < 1.0:
            proc_w = max(1, int(orig.shape[1] * self.resize_factor))
            proc_h = max(1, int(orig.shape[0] * self.resize_factor))
            proc = cv2.resize(orig, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
            scale_x = orig.shape[1] / proc_w
            scale_y = orig.shape[0] / proc_h
        else:
            proc = orig
            scale_x = scale_y = 1.0

        # YOLOv8 tracking with ByteTrack
        results = self.yolo.track(
            proc,
            persist=True,
            tracker=self.tracker_cfg,
            classes=[0],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )

        annotated = orig.copy()
        detections: List[Dict[str, Any]] = []

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            # Ultralytics Boxes: xyxy, id, cls, conf
            for i in range(len(boxes)):
                # class filter (person)
                if boxes.cls is not None and int(boxes.cls[i]) != 0:
                    continue

                track_id = -1
                if getattr(boxes, "id", None) is not None and boxes.id is not None:
                    try:
                        track_id = int(boxes.id[i])
                    except Exception:
                        track_id = -1

                x1 = int(round(float(boxes.xyxy[i][0]) * scale_x))
                y1 = int(round(float(boxes.xyxy[i][1]) * scale_y))
                x2 = int(round(float(boxes.xyxy[i][2]) * scale_x))
                y2 = int(round(float(boxes.xyxy[i][3]) * scale_y))

                x1 = max(0, min(orig.shape[1] - 1, x1))
                y1 = max(0, min(orig.shape[0] - 1, y1))
                x2 = max(x1 + 1, min(orig.shape[1], x2))
                y2 = max(y1 + 1, min(orig.shape[0], y2))

                crop = orig[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                identity_name = "Unknown"
                identity_conf = 0.0

                # Only run FaceNet occasionally per track to reduce load
                if track_id != -1 and not self._should_recognize(track_id):
                    cached = self._identity_cache.get(track_id)
                    if cached is not None:
                        identity_name, identity_conf = cached
                else:
                    try:
                        res = self.recognize_face_fn(crop, orig, (x1, y1, x2, y2))
                        if res:
                            identity_name = res.get("name", "Unknown") or "Unknown"
                            identity_conf = float(res.get("confidence", 0.0) or 0.0)
                    except Exception:
                        identity_name, identity_conf = "Unknown", 0.0

                    if track_id != -1:
                        self._last_recog_frame[track_id] = self._frame_idx
                        self._identity_cache[track_id] = (identity_name, identity_conf)

                auth_level = self.get_authorization_level(str(identity_name))
                color = self.get_authorization_color(auth_level)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"ID {track_id} | {identity_name} ({identity_conf:.2f})" if track_id != -1 else f"{identity_name} ({identity_conf:.2f})"
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 2, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    annotated,
                    auth_level,
                    (x1 + 2, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

                detections.append(
                    {
                        "track_id": track_id,
                        "identity": identity_name,
                        "authorization": auth_level,
                        "identity_conf": identity_conf,
                        "camera": "Primary",
                        "timestamp": time.strftime("%H:%M:%S"),
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        return annotated, detections
