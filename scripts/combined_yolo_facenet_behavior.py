"""Combined pipeline: YOLO + ByteTrack + FaceNet + Behavior Detection (HOI).

Merges face recognition with human-object interaction detection.
"""

from __future__ import annotations

import importlib.util
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


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
    behavior_status: str
    camera: str
    timestamp: str
    bbox: Tuple[int, int, int, int]


class CombinedYOLOFaceNetBehavior:
    """YOLOv8/YOLO26 + ByteTrack + FaceNet + Behavior Detection."""

    # Class IDs for office objects
    CLASS_ID_TO_NAME = {
        0: "person",
        24: "backpack",
        26: "handbag",
        63: "laptop",
        64: "mouse",
        66: "keyboard",
        67: "cell phone",
    }

    # Behavior detection rules
    CARRY_IDS = {24, 26, 67, 63}
    INTERACT_IDS = {63, 64, 66, 24, 26, 67}
    INTERACT_PRIORITY = [63, 66, 64, 67, 24, 26]

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
        enable_behavior: bool = True,
        coverage_thresh: float = 0.18,
        move_px_thresh: float = 8.0,
        stationary_frames_required: int = 6,
        status_on_frames_required: int = 6,
        status_off_frames_required: int = 12,
        object_hold_frames: int = 8,
    ) -> None:
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.resize_factor = float(resize_factor)
        self.frame_skip = max(int(frame_skip), 1)
        self.recog_interval = max(int(recog_interval), 1)
        self.tracker_cfg = str(tracker_cfg)
        self.device = device
        self.enable_behavior = bool(enable_behavior)

        # Behavior detection parameters
        self.coverage_thresh = float(coverage_thresh)
        self.move_px_thresh = float(move_px_thresh)
        self.stationary_frames_required = int(stationary_frames_required)
        self.status_on_frames_required = int(status_on_frames_required)
        self.status_off_frames_required = int(status_off_frames_required)
        self.object_hold_frames = int(object_hold_frames)

        self.authorization_map = authorization_map or {}

        # Load FaceNet module
        self.facenet_path = Path(facenet_main_path)
        if not self.facenet_path.is_absolute():
            self.facenet_path = (REPO_ROOT / self.facenet_path).resolve()
        if not self.facenet_path.exists():
            raise FileNotFoundError(f"FaceNet main not found: {self.facenet_path}")

        self.facenet_mod = load_module_from_path("facenet_main_behavior", self.facenet_path)
        self.recognize_face_fn = getattr(self.facenet_mod, "recognize_face_in_crop", None)
        if self.recognize_face_fn is None:
            raise AttributeError(
                f"{self.facenet_path.name} does not provide recognize_face_in_crop(person_crop, original_frame, person_bbox)"
            )

        # Load YOLO
        yolo_path = Path(yolo_model_path)
        if not yolo_path.is_absolute():
            yolo_path = (REPO_ROOT / yolo_path).resolve()
        if not yolo_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")

        from ultralytics import YOLO

        self.yolo = YOLO(str(yolo_path))

        # Track state for face recognition
        self._last_recog_frame: Dict[int, int] = {}
        self._identity_cache: Dict[int, Dict] = {}  # track_id -> {name, confidence, locked, lock_frame, last_face_frame}

        # Track state for behavior detection
        self.tracks: Dict[int, Dict] = {}
        self.last_objects: Dict[int, Dict] = {}

        self._frame_idx = 0
        
        # Identity locking parameters
        self.identity_lock_confidence = 0.7  # Confidence threshold to lock identity
        self.identity_lock_duration = 600  # Frames to keep locked identity (20 seconds at 30fps)
        self.identity_decay_rate = 0.998  # Confidence decay when face not detected

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

    @staticmethod
    def status_color(status_text: str):
        """Color for behavior status."""
        if "CARRYING" in status_text:
            return (0, 0, 255)  # Red
        if "INTERACTING" in status_text:
            return (0, 165, 255)  # Orange
        return (0, 255, 0)  # Green

    def _should_recognize(self, track_id: int) -> bool:
        last = self._last_recog_frame.get(track_id, -10_000)
        return (self._frame_idx - last) >= self.recog_interval

    def box_center_xyxy(self, xyxy):
        x1, y1, x2, y2 = xyxy
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    def update_object_cache(self, det_item_cls, det_item_xyxy, det_item_conf=None):
        """Cache objects for temporal smoothing."""
        for cid in list(self.last_objects.keys()):
            self.last_objects[cid]["miss"] += 1
            if self.last_objects[cid]["miss"] > self.object_hold_frames:
                del self.last_objects[cid]

        if det_item_conf is None:
            det_item_conf = np.ones(len(det_item_cls), dtype=np.float32)

        for cid, box, cf in zip(det_item_cls, det_item_xyxy, det_item_conf):
            cid = int(cid)
            cf = float(cf)
            if (cid not in self.last_objects) or (cf > float(self.last_objects[cid]["conf"])):
                self.last_objects[cid] = {"box": box.copy(), "miss": 0, "conf": cf}

    def get_held_objects(self):
        """Return cached objects as arrays."""
        if not self.last_objects:
            return np.array([], dtype=int), np.zeros((0, 4), dtype=np.float32)
        cids = np.fromiter(self.last_objects.keys(), dtype=int)
        boxes = np.stack([self.last_objects[cid]["box"] for cid in cids], axis=0).astype(np.float32)
        return cids, boxes

    def choose_status_for_person(self, moving: bool, stationary: bool, best_cov: Dict):
        """Determine behavior status based on movement and object overlap."""
        if moving:
            carry_candidates = [cid for cid in self.CARRY_IDS if cid in best_cov]
            if carry_candidates:
                chosen = max(carry_candidates, key=lambda cid: best_cov[cid][0])
                name = self.CLASS_ID_TO_NAME.get(chosen, str(chosen)).upper()
                return f"STATUS: CARRYING {name}"
            return "STATUS: NO INTERACTION"

        if stationary:
            interact_candidates = [cid for cid in self.INTERACT_IDS if cid in best_cov]
            if interact_candidates:
                chosen = None
                for cid in self.INTERACT_PRIORITY:
                    if cid in interact_candidates:
                        chosen = cid
                        break
                if chosen is None:
                    chosen = interact_candidates[0]
                name = self.CLASS_ID_TO_NAME.get(chosen, str(chosen)).upper()
                return f"STATUS: INTERACTING WITH {name}"

        return "STATUS: NO INTERACTION"

    def update_confirmed_status(self, track_state: Dict, candidate_status: str) -> str:
        """Apply hysteresis filtering to status changes."""
        if candidate_status == track_state["last_candidate"]:
            track_state["status_streak"] += 1
        else:
            track_state["last_candidate"] = candidate_status
            track_state["status_streak"] = 1

        if candidate_status == "STATUS: NO INTERACTION":
            if track_state["status_streak"] >= self.status_off_frames_required:
                track_state["confirmed_status"] = candidate_status
        else:
            if track_state["status_streak"] >= self.status_on_frames_required:
                track_state["confirmed_status"] = candidate_status

        return track_state["confirmed_status"]

    def process_frame(self, frame_bgr) -> Tuple[Any, List[Dict[str, Any]]]:
        """Process frame with face recognition and behavior detection.

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

        # Determine which classes to detect
        if self.enable_behavior:
            detect_classes = list(self.CLASS_ID_TO_NAME.keys())
        else:
            detect_classes = [0]  # Person only

        # YOLOv8 tracking with ByteTrack
        results = self.yolo.track(
            proc,
            persist=True,
            tracker=self.tracker_cfg,
            classes=detect_classes,
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

            # Extract all detections
            track_ids = []
            clss = []
            boxes_xyxy = []
            confs = []

            for i in range(len(boxes)):
                track_id = -1
                if getattr(boxes, "id", None) is not None and boxes.id is not None:
                    try:
                        track_id = int(boxes.id[i])
                    except Exception:
                        track_id = -1

                cls = int(boxes.cls[i])
                x1 = int(round(float(boxes.xyxy[i][0]) * scale_x))
                y1 = int(round(float(boxes.xyxy[i][1]) * scale_y))
                x2 = int(round(float(boxes.xyxy[i][2]) * scale_x))
                y2 = int(round(float(boxes.xyxy[i][3]) * scale_y))
                conf = float(boxes.conf[i])

                x1 = max(0, min(orig.shape[1] - 1, x1))
                y1 = max(0, min(orig.shape[0] - 1, y1))
                x2 = max(x1 + 1, min(orig.shape[1], x2))
                y2 = max(y1 + 1, min(orig.shape[0], y2))

                track_ids.append(track_id)
                clss.append(cls)
                boxes_xyxy.append([x1, y1, x2, y2])
                confs.append(conf)

            track_ids = np.array(track_ids)
            clss = np.array(clss)
            boxes_xyxy = np.array(boxes_xyxy)
            confs = np.array(confs)

            # Update object cache for behavior detection
            if self.enable_behavior:
                det_item_mask = (clss != 0)
                self.update_object_cache(clss[det_item_mask], boxes_xyxy[det_item_mask], confs[det_item_mask])
                item_cls, item_xyxy = self.get_held_objects()
            else:
                item_cls = np.array([], dtype=int)
                item_xyxy = np.zeros((0, 4), dtype=np.float32)

            # Process persons
            for i, (tid, cls, bbox, conf) in enumerate(zip(track_ids, clss, boxes_xyxy, confs)):
                if cls != 0:  # Not a person
                    continue

                x1, y1, x2, y2 = bbox
                crop = orig[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # === Face Recognition ===
                identity_name = "Unknown"
                identity_conf = 0.0

                # Initialize identity cache for new tracks
                if tid != -1 and tid not in self._identity_cache:
                    self._identity_cache[tid] = {
                        "name": "Unknown",
                        "confidence": 0.0,
                        "locked": False,
                        "lock_frame": 0,
                        "last_face_frame": 0,
                        "frames_without_face": 0,
                    }

                # Try face recognition (throttled per track)
                if tid != -1 and self._should_recognize(tid):
                    try:
                        res = self.recognize_face_fn(crop, orig, (x1, y1, x2, y2))
                        if res:
                            detected_name = res.get("name", "Unknown") or "Unknown"
                            detected_conf = float(res.get("confidence", 0.0) or 0.0)
                            
                            # Update identity cache with locking logic
                            cache = self._identity_cache[tid]
                            
                            if detected_name != "Unknown":
                                # Valid face detected
                                if detected_conf >= self.identity_lock_confidence:
                                    # High confidence - lock identity
                                    cache["name"] = detected_name
                                    cache["confidence"] = detected_conf
                                    cache["locked"] = True
                                    cache["lock_frame"] = self._frame_idx
                                    cache["last_face_frame"] = self._frame_idx
                                    cache["frames_without_face"] = 0
                                elif detected_name == cache["name"]:
                                    # Same person confirmation - boost confidence
                                    cache["confidence"] = min(1.0, cache["confidence"] * 1.02)
                                    cache["last_face_frame"] = self._frame_idx
                                    cache["frames_without_face"] = 0
                                    if cache["confidence"] >= self.identity_lock_confidence:
                                        cache["locked"] = True
                                        cache["lock_frame"] = self._frame_idx
                                elif not cache["locked"]:
                                    # Different person, not locked - update
                                    cache["name"] = detected_name
                                    cache["confidence"] = detected_conf
                                    cache["last_face_frame"] = self._frame_idx
                                    cache["frames_without_face"] = 0
                            else:
                                # Face not detected (Unknown)
                                cache["frames_without_face"] += 1
                                
                                # Maintain locked identity with decay
                                if cache["locked"]:
                                    frames_since_lock = self._frame_idx - cache["lock_frame"]
                                    
                                    # Keep locked identity for configured duration
                                    if frames_since_lock < self.identity_lock_duration:
                                        # Apply confidence decay
                                        cache["confidence"] *= self.identity_decay_rate
                                        # Maintain minimum confidence while locked
                                        cache["confidence"] = max(cache["confidence"], 0.3)
                                    else:
                                        # Lock expired
                                        cache["locked"] = False
                                        cache["confidence"] *= 0.9
                                else:
                                    # Not locked, decay faster
                                    cache["confidence"] *= 0.95
                                    if cache["confidence"] < 0.15:
                                        cache["name"] = "Unknown"
                                        cache["confidence"] = 0.0
                            
                            self._last_recog_frame[tid] = self._frame_idx
                        else:
                            # Recognition function returned None
                            if tid in self._identity_cache:
                                cache = self._identity_cache[tid]
                                cache["frames_without_face"] += 1
                                if cache["locked"]:
                                    cache["confidence"] *= self.identity_decay_rate
                                    cache["confidence"] = max(cache["confidence"], 0.3)
                                else:
                                    cache["confidence"] *= 0.95
                    except Exception as e:
                        # Recognition error - maintain cache with decay
                        if tid in self._identity_cache:
                            cache = self._identity_cache[tid]
                            cache["frames_without_face"] += 1
                            if cache["locked"]:
                                cache["confidence"] *= self.identity_decay_rate
                                cache["confidence"] = max(cache["confidence"], 0.3)
                            else:
                                cache["confidence"] *= 0.95

                # Get identity from cache
                if tid != -1 and tid in self._identity_cache:
                    cache = self._identity_cache[tid]
                    identity_name = cache["name"]
                    identity_conf = cache["confidence"]

                auth_level = self.get_authorization_level(str(identity_name))
                auth_color = self.get_authorization_color(auth_level)

                # === Behavior Detection ===
                # Only apply behavior detection for "Partially Authorized" persons
                behavior_status = "STATUS: NO INTERACTION"

                if self.enable_behavior and tid != -1 and auth_level == "Partially Authorized":
                    # Initialize track state
                    if tid not in self.tracks:
                        self.tracks[tid] = {
                            "center": self.box_center_xyxy(bbox),
                            "still_frames": 0,
                            "speed": 0.0,
                            "last_candidate": "STATUS: NO INTERACTION",
                            "status_streak": 0,
                            "confirmed_status": "STATUS: NO INTERACTION",
                        }

                    st = self.tracks[tid]
                    p_center = self.box_center_xyxy(bbox)
                    st["speed"] = float(np.linalg.norm(p_center - st["center"]))
                    st["center"] = p_center

                    moving = st["speed"] >= self.move_px_thresh
                    st["still_frames"] = 0 if moving else (st["still_frames"] + 1)
                    stationary = st["still_frames"] >= self.stationary_frames_required

                    # Vectorized IoA calculation
                    best_cov = {}
                    if len(item_xyxy) > 0:
                        p_box = bbox
                        ix1 = np.maximum(p_box[0], item_xyxy[:, 0])
                        iy1 = np.maximum(p_box[1], item_xyxy[:, 1])
                        ix2 = np.minimum(p_box[2], item_xyxy[:, 2])
                        iy2 = np.minimum(p_box[3], item_xyxy[:, 3])

                        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                        item_areas = (item_xyxy[:, 2] - item_xyxy[:, 0]) * (item_xyxy[:, 3] - item_xyxy[:, 1])
                        ioas = inter / (item_areas + 1e-6)

                        for idx in np.where(ioas > self.coverage_thresh)[0]:
                            o_cls = int(item_cls[idx])
                            if o_cls not in best_cov or ioas[idx] > best_cov[o_cls][0]:
                                best_cov[o_cls] = (ioas[idx], item_xyxy[idx])

                    # Update status
                    candidate = self.choose_status_for_person(moving, stationary, best_cov)
                    behavior_status = self.update_confirmed_status(st, candidate)

                # === Draw Annotations ===
                # Draw bounding box (color based on authorization)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), auth_color, 2)

                # Line 1: Behavior status (if enabled and Partially Authorized) - at the TOP
                y_offset = y1 - 15
                if self.enable_behavior and auth_level == "Partially Authorized" and behavior_status != "STATUS: NO INTERACTION":
                    status_color = self.status_color(behavior_status)
                    cv2.putText(
                        annotated,
                        behavior_status,
                        (x1 + 2, max(20, y_offset)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        status_color,
                        2,
                    )
                    y_offset -= 18  # Move up for next line

                # Line 2: Track ID + Identity
                label = f"ID {tid} | {identity_name} ({identity_conf:.2f})" if tid != -1 else f"{identity_name} ({identity_conf:.2f})"
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 2, max(35, y_offset)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                y_offset -= 15  # Move up for next line

                # Line 3: Authorization level
                cv2.putText(
                    annotated,
                    auth_level,
                    (x1 + 2, max(50, y_offset)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    auth_color,
                    1,
                )

                # Add to detections
                detections.append(
                    {
                        "track_id": tid,
                        "identity": identity_name,
                        "authorization": auth_level,
                        "identity_conf": identity_conf,
                        "behavior_status": behavior_status,
                        "camera": "Primary",
                        "timestamp": time.strftime("%H:%M:%S"),
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        return annotated, detections
