from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any


def iou(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


class PersonTracker:
    """Lightweight greedy IoU tracker for person bounding boxes.
    Not a Kalman-filter SORT, but stabilizes boxes and assigns persistent IDs.
    """
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    def update(self, detections: List[Tuple[int, int, int, int]]):
        """Update tracks with new detections. Returns list of current tracks.
        detections: list of (x1,y1,x2,y2)
        """
        dets = [tuple(map(int, d)) for d in detections]

        if len(self.tracks) == 0:
            # create tracks for all detections
            for d in dets:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = {"bbox": d, "age": 0, "time_since_update": 0, "hits": 1}
            return [{"track_id": k, "bbox": v["bbox"]} for k, v in self.tracks.items()]

        # Build IOU matrix between existing tracks and detections
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(dets)), dtype=float)
        for i, tid in enumerate(track_ids):
            for j, d in enumerate(dets):
                iou_matrix[i, j] = iou(self.tracks[tid]["bbox"], d)

        matched_tracks = set()
        matched_dets = set()
        # Greedy matching by highest IoU
        while True:
            if iou_matrix.size == 0:
                break
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[i, j]
            if max_iou < self.iou_threshold:
                break
            tid = track_ids[i]
            self.tracks[tid]["bbox"] = dets[j]
            self.tracks[tid]["time_since_update"] = 0
            self.tracks[tid]["age"] = 0
            self.tracks[tid]["hits"] += 1
            matched_tracks.add(i)
            matched_dets.add(j)
            # invalidate row i and col j
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1

        # unmatched detections -> create new tracks
        for j, d in enumerate(dets):
            if j in matched_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = {"bbox": d, "age": 0, "time_since_update": 0, "hits": 1}

        # unmatched tracks -> age and potentially delete
        for i, tid in enumerate(track_ids):
            if i in matched_tracks:
                continue
            self.tracks[tid]["time_since_update"] += 1
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["time_since_update"] > self.max_age:
                del self.tracks[tid]

        return [{"track_id": k, "bbox": v["bbox"]} for k, v in self.tracks.items()]


if __name__ == "__main__":
    print("PersonTracker module")
