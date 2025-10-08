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


class SortLikeTracker:
    """A lightweight SORT-like tracker using constant velocity prediction and greedy IoU matching.
    - prediction: each track keeps a bbox and a velocity vector; predict next bbox by adding velocity.
    - matching: greedy highest-IoU matching between predicted bboxes and detections.
    This is not a full Kalman implementation but improves stability over raw detection jitter.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    @staticmethod
    def _to_xywh(box):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        return np.array([cx, cy, w, h], dtype=float)

    @staticmethod
    def _to_box(xywh):
        cx, cy, w, h = xywh
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return (int(x1), int(y1), int(x2), int(y2))

    def predict(self):
        # predict next state for each track by adding velocity
        for tid, t in list(self.tracks.items()):
            state = t["state"]
            vel = t.get("vel", np.zeros(4, dtype=float))
            pred = state + vel
            t["pred"] = pred

    def update(self, detections: List[Tuple[int, int, int, int]]):
        dets = [tuple(map(int, d)) for d in detections]

        # predict step
        if len(self.tracks) > 0:
            self.predict()

        # build predicted boxes list
        pred_list = [self._to_box(t.get("pred", self._to_xywh(t["bbox"]))) for t in self.tracks.values()]
        track_ids = list(self.tracks.keys())

        # IoU matrix between predicted tracks and detections
        if len(pred_list) > 0 and len(dets) > 0:
            iou_mat = np.zeros((len(pred_list), len(dets)), dtype=float)
            for i, pb in enumerate(pred_list):
                for j, db in enumerate(dets):
                    iou_mat[i, j] = iou(pb, db)
        else:
            iou_mat = np.zeros((len(pred_list), len(dets)), dtype=float)

        matched_tracks = set()
        matched_dets = set()

        # greedy matching by max IoU
        while iou_mat.size > 0:
            idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            i, j = idx
            if iou_mat[i, j] < self.iou_threshold:
                break
            tid = track_ids[i]
            # update track with detection j
            det_xywh = self._to_xywh(dets[j])
            prev_state = self.tracks[tid]["state"]
            vel = det_xywh - prev_state
            self.tracks[tid]["state"] = det_xywh
            self.tracks[tid]["vel"] = vel
            self.tracks[tid]["bbox"] = dets[j]
            self.tracks[tid]["time_since_update"] = 0
            self.tracks[tid]["hits"] += 1
            matched_tracks.add(i)
            matched_dets.add(j)
            # invalidate row i and col j
            iou_mat[i, :] = -1
            iou_mat[:, j] = -1

        # create new tracks for unmatched detections
        for j, d in enumerate(dets):
            if j in matched_dets:
                continue
            xywh = self._to_xywh(d)
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = {
                "bbox": d,
                "state": xywh,
                "vel": np.zeros(4, dtype=float),
                "hits": 1,
                "time_since_update": 0,
                "age": 0,
            }

        # age unmatched tracks and remove old ones
        for idx, tid in enumerate(track_ids):
            if idx in matched_tracks:
                continue
            t = self.tracks.get(tid)
            if t is None:
                continue
            t["time_since_update"] += 1
            t["age"] += 1
            if t["time_since_update"] > self.max_age:
                del self.tracks[tid]

        # return list of active tracks
        return [{"track_id": k, "bbox": v["bbox"]} for k, v in self.tracks.items()]


if __name__ == "__main__":
    print("SortLikeTracker module")
