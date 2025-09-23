from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
from collections import deque, Counter

# Minimal recognition tracker (copy of previous implementation)

def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
    area2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def point_in_box(point: Tuple[int, int], box: Tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


class RecognitionTracker:
    def __init__(self, cosine_threshold: float = 0.60, unknown_ttl: int = 30, max_embeddings_per_person: int = 20, iou_threshold: float = 0.3,
                 smoothing_window: int = 5, promotion_fraction: float = 0.6,
                 decay_half_life: int = 90, display_threshold: float = 0.5, delete_threshold: float = 0.05):
        """
        smoothing_window: number of recent name observations to keep per tracked person
        promotion_fraction: fraction of votes required among the window to change the stable name
        """
        self.cosine_threshold = cosine_threshold
        self.unknown_ttl = unknown_ttl
        self.max_embeddings_per_person = max_embeddings_per_person
        self.iou_threshold = iou_threshold
        # smoothing parameters
        self.smoothing_window = smoothing_window
        self.promotion_fraction = promotion_fraction
        # Decay parameters (exponential decay per frame)
        # decay_half_life: number of frames for confidence to halve (e.g., 90 ~3s at 30fps)
        self.decay_half_life = max(1, int(decay_half_life))
        # alpha per frame such that confidence *= alpha each frame without evidence
        self.decay_alpha = float(0.5 ** (1.0 / float(self.decay_half_life)))
        self.display_threshold = float(display_threshold)
        self.delete_threshold = float(delete_threshold)

        self.tracked_persons: Dict[int, Dict[str, Any]] = {}
        self.next_person_id = 1
        self.unknown_trackers: Dict[int, Dict[str, Any]] = {}
        self.next_unknown_id = 1

    @staticmethod
    def normalize_embedding(emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def expire_old(self, frame_num: int) -> None:
        # Old expiration still in place but we'll also remove based on decayed confidence
        to_del = [pid for pid, d in self.tracked_persons.items() if frame_num - d.get("last_seen", frame_num) > self.unknown_ttl]
        for pid in to_del:
            del self.tracked_persons[pid]
        to_del_u = [uid for uid, d in self.unknown_trackers.items() if frame_num - d.get("last_seen", frame_num) > self.unknown_ttl]
        for uid in to_del_u:
            del self.unknown_trackers[uid]

    def process_frame(self, people_boxes: List[Tuple[int, int, int, int]], faces: List[Dict[str, Any]], embeddings: Optional[np.ndarray], face_to_person: Dict[int, int], frame_num: int):
        emb_list = []
        if embeddings is not None and len(embeddings) > 0:
            arr = np.asarray(embeddings)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            emb_list = [self.normalize_embedding(e) for e in arr]
        else:
            emb_list = [None] * len(faces)

        self.expire_old(frame_num)

        # Apply exponential decay to confidence for tracks that were NOT updated this frame.
        # We'll reduce confidence by decay_alpha for each track whose last_seen != frame_num.
        tracks_to_remove = []
        for pid, d in list(self.tracked_persons.items()):
            if d.get("last_seen", frame_num) != frame_num:
                # decay confidence if present, else create a small default
                cur_conf = float(d.get("confidence", 1.0))
                new_conf = cur_conf * self.decay_alpha
                d["confidence"] = new_conf
            # If confidence is very low and not seen for a while, schedule removal
            if d.get("confidence", 0.0) <= self.delete_threshold and frame_num - d.get("last_seen", frame_num) > (self.decay_half_life * 4):
                tracks_to_remove.append(pid)
        for pid in tracks_to_remove:
            del self.tracked_persons[pid]

        # Same decay for unknown trackers (keep embeddings but decay any pseudo-confidence)
        u_to_remove = []
        for uid, ud in list(self.unknown_trackers.items()):
            if ud.get("last_seen", frame_num) != frame_num:
                ud["confidence"] = float(ud.get("confidence", 0.0)) * self.decay_alpha
            if ud.get("confidence", 0.0) <= self.delete_threshold and frame_num - ud.get("last_seen", frame_num) > (self.decay_half_life * 4):
                u_to_remove.append(uid)
        for uid in u_to_remove:
            del self.unknown_trackers[uid]

        def find_tracked_for_box(box: Tuple[int,int,int,int]):
            best = None
            best_iou = 0.0
            for pid, d in self.tracked_persons.items():
                iou = compute_iou(box, d["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best = pid
            return best

        for i, face in enumerate(faces):
            emb = emb_list[i]
            name = face.get("name", "Unknown")
            prob = float(face.get("prob", 0.0) or 0.0)
            if name != "Unknown":
                person_box_idx = face_to_person.get(i)
                attached_pid = None
                if person_box_idx is not None and 0 <= person_box_idx < len(people_boxes):
                    attached_pid = find_tracked_for_box(people_boxes[person_box_idx])
                if attached_pid is None:
                    best_pid = None
                    best_sim = -1.0
                    if emb is not None:
                        for pid, d in self.tracked_persons.items():
                            if d.get("embeddings"):
                                centroid = np.mean(d["embeddings"], axis=0)
                                centroid = self.normalize_embedding(centroid)
                                sim = self.cosine_sim(emb, centroid)
                                if sim > best_sim:
                                    best_sim = sim
                                    best_pid = pid
                    if best_sim >= self.cosine_threshold:
                        attached_pid = best_pid
                if attached_pid is not None:
                    d = self.tracked_persons[attached_pid]
                    if person_box_idx is not None and 0 <= person_box_idx < len(people_boxes):
                        d["bbox"] = people_boxes[person_box_idx]
                    d["last_seen"] = frame_num
                    d["confidence"] = prob
                    # record raw name (most recent) and update smoothing votes
                    d["name"] = name
                    votes = d.setdefault("name_votes", deque(maxlen=self.smoothing_window))
                    votes.append(name)
                    # recompute stable_name if votes exceed threshold
                    cnt = Counter(votes)
                    most_common, count = cnt.most_common(1)[0]
                    if count >= int(self.promotion_fraction * len(votes)):
                        d["stable_name"] = most_common
                    if emb is not None:
                        d.setdefault("embeddings", []).append(emb)
                        if len(d["embeddings"]) > self.max_embeddings_per_person:
                            d["embeddings"] = d["embeddings"][-self.max_embeddings_per_person:]
                else:
                    bbox = people_boxes[person_box_idx] if (person_box_idx is not None and 0 <= person_box_idx < len(people_boxes)) else face.get("bbox")
                    pid = self.next_person_id
                    self.next_person_id += 1
                    self.tracked_persons[pid] = {
                        "bbox": bbox,
                        "name": name,
                        "stable_name": name if name != "Unknown" else None,
                        "name_votes": deque([name], maxlen=self.smoothing_window),
                        "last_seen": frame_num,
                        "confidence": prob,
                        "embeddings": [emb] if emb is not None else [],
                    }
                    # promote unknown trackers if overlapped
                    to_promote = []
                    for uid, ud in list(self.unknown_trackers.items()):
                        if compute_iou(bbox, ud["bbox"]) >= self.iou_threshold:
                            to_promote.append(uid)
                    for uid in to_promote:
                        ud = self.unknown_trackers.pop(uid)
                        if ud.get("embeddings"):
                            self.tracked_persons[pid].setdefault("embeddings", []).extend(ud["embeddings"])                    
                        # if unknown carried votes (rare), merge into name_votes
                        if ud.get("name_votes"):
                            votes = self.tracked_persons[pid].setdefault("name_votes", deque(maxlen=self.smoothing_window))
                            for v in ud["name_votes"]:
                                votes.append(v)
                            cnt = Counter(votes)
                            most_common, count = cnt.most_common(1)[0]
                            if count >= int(self.promotion_fraction * len(votes)):
                                self.tracked_persons[pid]["stable_name"] = most_common
        for i, face in enumerate(faces):
            emb = emb_list[i]
            name = face.get("name", "Unknown")
            if name == "Unknown":
                fb = face.get("bbox")
                face_center = ((fb[0] + fb[2]) // 2, (fb[1] + fb[3]) // 2)
                belongs_to_known = False
                for pid, d in self.tracked_persons.items():
                    if point_in_box(face_center, d["bbox"]):
                        belongs_to_known = True
                        break
                if belongs_to_known:
                    continue
                matched_uid = None
                person_box_idx = face_to_person.get(i)
                if person_box_idx is not None and 0 <= person_box_idx < len(people_boxes):
                    pbox = people_boxes[person_box_idx]
                    for uid, ud in self.unknown_trackers.items():
                        if compute_iou(pbox, ud["bbox"]) >= self.iou_threshold:
                            matched_uid = uid
                            break
                if matched_uid is None and emb is not None:
                    best_uid = None
                    best_sim = -1.0
                    for uid, ud in self.unknown_trackers.items():
                        if ud.get("embeddings"):
                            centroid = np.mean(ud["embeddings"], axis=0)
                            centroid = self.normalize_embedding(centroid)
                            sim = self.cosine_sim(emb, centroid)
                            if sim > best_sim:
                                best_sim = sim
                                best_uid = uid
                    if best_sim >= self.cosine_threshold:
                        matched_uid = best_uid
                if matched_uid is not None:
                    ud = self.unknown_trackers[matched_uid]
                    if person_box_idx is not None and 0 <= person_box_idx < len(people_boxes):
                        ud["bbox"] = people_boxes[person_box_idx]
                    ud["last_seen"] = frame_num
                    if emb is not None:
                        ud.setdefault("embeddings", []).append(emb)
                        if len(ud["embeddings"]) > self.max_embeddings_per_person:
                            ud["embeddings"] = ud["embeddings"][-self.max_embeddings_per_person:]
                    # carry name votes for unknowns (in case classifier later provides a label)
                    ud.setdefault("name_votes", deque(maxlen=self.smoothing_window))
                    # unknown entries don't add a label now
                else:
                    bbox = people_boxes[person_box_idx] if (person_box_idx is not None and 0 <= person_box_idx < len(people_boxes)) else face.get("bbox")
                    uid = self.next_unknown_id
                    self.next_unknown_id += 1
                    self.unknown_trackers[uid] = {
                        "bbox": bbox,
                        "last_seen": frame_num,
                        "embeddings": [emb] if emb is not None else [],
                        "name_votes": deque(maxlen=self.smoothing_window),
                    }
        unknown_person_boxes = set()
        for i, face in enumerate(faces):
            if face.get("name") == "Unknown":
                fb = face.get("bbox")
                face_center = ((fb[0] + fb[2]) // 2, (fb[1] + fb[3]) // 2)
                belongs_to_known = False
                for pid, d in self.tracked_persons.items():
                    if point_in_box(face_center, d["bbox"]):
                        belongs_to_known = True
                        break
                if belongs_to_known:
                    continue
                if i in face_to_person:
                    pb_idx = face_to_person[i]
                    # pb_idx is an index into people_boxes; check if this person box overlaps any tracked person bbox
                    try:
                        pbox = people_boxes[pb_idx]
                    except Exception:
                        pbox = None
                    if pbox is not None:
                        overlaps_tracked = False
                        for pid, d in self.tracked_persons.items():
                            if compute_iou(pbox, d.get("bbox", (0,0,0,0))) >= self.iou_threshold:
                                overlaps_tracked = True
                                break
                        if not overlaps_tracked:
                            unknown_person_boxes.add(pb_idx)
        face_display_names = []
        for f in faces:
            name = f.get("name", "Unknown")
            fb = f.get("bbox")
            face_center = ((fb[0] + fb[2]) // 2, (fb[1] + fb[3]) // 2)
            assigned_name = name
            if name == "Unknown":
                for pid, d in self.tracked_persons.items():
                    if point_in_box(face_center, d["bbox"]):
                        # prefer stable_name to reduce flicker; fall back to most recent name
                        assigned_name = d.get("stable_name") or d.get("name")
                        break
            face_display_names.append(assigned_name)
        return self.tracked_persons, unknown_person_boxes, face_display_names


if __name__ == "__main__":
    print('recognition_tracker module')
